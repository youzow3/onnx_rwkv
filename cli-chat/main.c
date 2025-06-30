
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <onnxruntime/onnxruntime_c_api.h>
#include <wchar.h>

char *g_args_name[] = { "help", "verbose", "model", "tokenizer" };

enum ArgsIds
{
  ARGS_HELP,
  ARGS_VERBOSE,
  ARGS_MODEL,
  ARGS_TOKENIZER,
  ARGS_MAX,
  ARGS_NONE = -1
};

struct Args
{
  bool help;
  bool verbose;
  char *model;
  char *tokenizer;
} g_args = { 0 };

struct RWKV
{
  OrtSession *session;
  OrtSessionOptions *session_options;
  OrtRunOptions *run_options;

  size_t ninput;
  size_t noutput;

  char **input_name;
  char **output_name;

  size_t *ndim;
  int64_t **dim;
  const char ***axis_name;
};

struct RWKVTokenizerEntry
{
  struct RWKVTokenizerEntry *next[256];
  struct RWKVTokenizerEntry *prev;
  int64_t id;
  int64_t ch; // char ch;
};

struct RWKVTokenizer
{
  struct RWKVTokenizerEntry root;
  const char **token2text;
  size_t vocab_size;
};

const OrtApi *g_ort_api;
OrtEnv *g_ort_env;
OrtAllocator *g_ort_allocator;

#define info_printf(...)                                                      \
  {                                                                           \
    if (g_args.verbose)                                                       \
      fprintf (stderr, "I: " __VA_ARGS__);                                    \
  }

#define warning_printf(...)                                                   \
  {                                                                           \
    fprintf (stderr, "W: ", __VA_ARGS__);                                     \
  }

#define error_printf(...)                                                     \
  {                                                                           \
    fprintf (stderr, "E: " __VA_ARGS__);                                      \
  }

#define check_status_and_abort(status)                                        \
  if (status)                                                                 \
    {                                                                         \
      fprintf (stderr, "%s\n", g_ort_api->GetErrorMessage (status));          \
    }

static bool
parse_args (int argc, char **argv)
{
  int args_id = ARGS_NONE;
  int args_id_next = ARGS_NONE;

  for (size_t k = 1; k < argc; k++)
    {
      switch (args_id)
        {
        case ARGS_MODEL:
          g_args.model = argv[k];
          break;
        case ARGS_TOKENIZER:
          g_args.tokenizer = argv[k];
          break;
        default:
          if (!strcmp (argv[k], "-h") || !strcmp (argv[k], "--help"))
            g_args.help = true;
          if (!strcmp (argv[k], "-v") || !strcmp (argv[k], "--verbose"))
            g_args.verbose = true;
          if (!strcmp (argv[k], "-m") || !strcmp (argv[k], "--model"))
            args_id_next = ARGS_MODEL;
          if (!strcmp (argv[k], "-t") || !strcmp (argv[k], "--tokenizer"))
            args_id_next = ARGS_TOKENIZER;
        }
      args_id = args_id_next;
      args_id_next = ARGS_NONE;
    }

  return args_id == ARGS_NONE;
}

static bool
is_valid_args (struct Args *args)
{
  if (args->model == NULL)
    return false;
  if (args->tokenizer == NULL)
    return false;
  return true;
}

static void
help (void)
{
  printf ("Usage: command [options]\n"
          "-h --help      \tDisplay this information.\n"
          "-v --verbose   \tVerbose output.\n"
          "-m --model     \tSpecify model used in this program.\n"
          "-t --tokenizer \tSpecify tokenizer used in this program.\n");
}

static size_t
u8len (char *ch, size_t len)
{
  size_t estimate_size = 1;
  if ((ch[0] & 0xf8) == 0xf0)
    estimate_size = 4;
  if ((ch[0] & 0xf0) == 0xe0)
    estimate_size = 3;
  if ((ch[0] & 0xe0) == 0xc0)
    estimate_size = 2;
  if ((ch[0] & 0x80) == 0)
    return 1;

  for (size_t k = 1; k < estimate_size; k++)
    if ((ch[k] & 0xc0) != 0x80)
      return (size_t)-1;
  return estimate_size;
}

static size_t
u8slen (char *str)
{
  size_t alen = strlen (str);
  size_t ulen = 0;
  for (size_t k = 0; k < alen;)
    {
      size_t uclen = u8len (str + k, alen - k);
      if (uclen == (size_t)-1)
        return -1;
      k += uclen;
      ulen++;
    }

  return ulen;
}

static size_t
itou8 (uint32_t i, char *u8, size_t u8size)
{
  if (u8size == 0)
    return 0;

  if (i < 0x80)
    {
      u8[0] = i;
      return 1;
    }

  if (i < 0x800)
    {
      if (u8size < 2)
        return 0;
      u8[0] = 0xc0 | (i >> 6);
      u8[1] = 0x80 | (i & 0x3f);
      return 2;
    }

  if (i < 0x10000)
    {
      if (u8size < 3)
        return 0;
      u8[0] = 0xe0 | (i >> 12);
      u8[1] = 0x80 | ((i >> 6) & 0x3f);
      u8[2] = 0x80 | (i & 0x3f);
      return 3;
    }

  if (i < 0x110000)
    {
      if (u8size < 4)
        return 0;
      u8[0] = 0xf0 | (i >> 18);
      u8[1] = 0x80 | ((i >> 12) & 0x3f);
      u8[2] = 0x80 | ((i >> 6) & 0x3f);
      u8[3] = 0x80 | (i & 0x3f);
      return 4;
    }

  return 0;
}

static char *
python_str (char *str, size_t *py_str_len)
{
  char *squote = strchr (str, '\'');
  char *dquote = strchr (str, '\"');
  char quote = 0;
  char *quote_pos = NULL;
  if (squote == NULL)
    {
      quote = '\"';
      quote_pos = dquote;
    }
  else if (dquote == NULL)
    {
      quote = '\'';
      quote_pos = squote;
    }
  else if ((squote != NULL) && (dquote != NULL))
    {
      quote = (squote - dquote) < 0 ? '\'' : '\"';
      quote_pos = (squote - dquote) < 0 ? squote : dquote;
    }
  else
    return NULL;

  bool raw = false;
  bool unicode = false;
  bool format = false;
  bool bytes = false;
  for (size_t k = 0; k < (quote_pos - str); k++)
    {
      switch (tolower (str[k]))
        {
        case 'r':
          raw = true;
          break;
        case 'u':
          unicode = true;
          break;
        case 'f':
          format = true;
          break;
        case 'b':
          bytes = true;
          break;
        }
    }

  if (format)
    {
      error_printf ("f-string is not supported.\n");
      return NULL;
    }

  size_t quote_pos_len = strlen (quote_pos);
  size_t encoded_str_size = quote_pos_len;
  char *encoded_str = malloc (encoded_str_size);
  if (encoded_str == NULL)
    return NULL;
  size_t encoded_str_pos = 0;
  bool escape = false;
  char escape_str[16];
  size_t escape_str_pos = 0;
  for (size_t k = 1; k < quote_pos_len;)
    {
      size_t ch_len = u8len (quote_pos + k, quote_pos_len - k);
      if (ch_len == -1)
        {
          error_printf ("invalid UTF-8 sequence.\n");
          goto on_encode_error;
        }

      if ((ch_len != 1) && escape)
        {
          error_printf ("Invalid escape sequence.\n");
          goto on_encode_error;
        }

      if (escape && (escape_str_pos > 0))
        {
          switch (escape_str[0])
            {
            case '\\':
            case '\'':
            case '\"':
              encoded_str[encoded_str_pos++] = escape_str[0];
              escape = false;
              break;
            case 'a':
              encoded_str[encoded_str_pos++] = '\a';
              escape = false;
              break;
            case 'b':
              encoded_str[encoded_str_pos++] = '\b';
              escape = false;
              break;
            case 'f':
              encoded_str[encoded_str_pos++] = '\f';
              escape = false;
              break;
            case 'n':
              encoded_str[encoded_str_pos++] = '\n';
              escape = false;
              break;
            case 'r':
              encoded_str[encoded_str_pos++] = '\r';
              escape = false;
              break;
            case 't':
              encoded_str[encoded_str_pos++] = '\t';
              escape = false;
              break;
            case 'v':
              encoded_str[encoded_str_pos++] = '\v';
              escape = false;
              break;
            case 'u':
            case 'U':
              if (bytes)
                {
                  error_printf (
                      "Unicode code points are not supported for bytes.\n");
                  goto on_encode_error;
                }
            case 'o':
            case 'x':
              escape = isxdigit (quote_pos[k]);
              break;
            case 'N':
              error_printf ("Not supported escape sequence.\n");
              goto on_encode_error;
            }

          if (!escape && (escape_str[0] == 'o'))
            {
              int code = strtol (escape_str + 1, NULL, 8);
              if (bytes)
                encoded_str[encoded_str_pos++] = code;
              else
                {
                  size_t ul = itou8 (code, encoded_str,
                                     encoded_str_size - encoded_str_pos);
                  encoded_str_pos += ul;
                }
            }
          if (!escape && (escape_str[0] == 'x'))
            {
              int code = strtol (escape_str + 1, NULL, 16);
              if (bytes)
                encoded_str[encoded_str_pos++] = code;
              else
                {
                  size_t ul = itou8 (code, encoded_str,
                                     encoded_str_size - encoded_str_pos);
                  encoded_str_pos += ul;
                }
            }
          if (!escape && (escape_str[0] == 'u'))
            {
              if (strlen (escape_str + 1) != 4)
                {
                  error_printf ("\\u need exactly 4 digits.\n");
                  goto on_encode_error;
                }
              size_t l = itou8 (strtol (escape_str + 1, NULL, 16), encoded_str,
                                encoded_str_size - encoded_str_pos);
              if (l == 0)
                {
                  error_printf ("Encoding error occured.\n");
                  goto on_encode_error;
                }
              encoded_str_pos += l;
            }
          if (!escape && (escape_str[0] == 'U'))
            {
              if (strlen (escape_str + 1) != 8)
                {
                  error_printf ("\\u need exactly 8 digits.\n");
                  goto on_encode_error;
                }
              size_t l = itou8 (strtol (escape_str + 1, NULL, 16), encoded_str,
                                encoded_str_size - encoded_str_pos);
              if (l == 0)
                {
                  error_printf ("Encoding error occured.\n");
                  goto on_encode_error;
                }
              encoded_str_pos += l;
            }
        }

      if (!escape && (quote_pos[k] == quote))
        {
          encoded_str[encoded_str_pos] = 0;
          break;
        }

      if (!raw && !escape && (quote_pos[k] == '\\'))
        {
          escape = true;
          escape_str_pos = 0;
          memset (escape_str, 0, sizeof (escape_str));
          k += ch_len;
          continue;
        }

      if (!escape || raw)
        {
          memcpy (encoded_str + encoded_str_pos, quote_pos + k, ch_len);
          encoded_str_pos += ch_len;
          k += ch_len;
          continue;
        }

      escape_str[escape_str_pos++] = quote_pos[k++];
    }

  *py_str_len = encoded_str_pos;
  return encoded_str;
on_encode_error:
  free (encoded_str);
  return NULL;
}

static bool
rwkv_tokenizer_init (struct RWKVTokenizer *tokenizer, char *filename)
{
  FILE *file = fopen (filename, "rt");
  if (file == NULL)
    return false;

  memset (&tokenizer->root, 0, sizeof (struct RWKVTokenizerEntry));
  tokenizer->root.ch = 0;
  tokenizer->root.id = -1;

  size_t vocab_buf_chunk = 65536;
  size_t vocab_buf_size = vocab_buf_chunk;
  size_t vocab_size = 0;
  tokenizer->token2text = malloc (sizeof (char *) * vocab_buf_chunk);
  if (tokenizer->token2text == NULL)
    goto on_error;
  memset (tokenizer->token2text, 0, sizeof (char *) * vocab_buf_chunk);

  char buf[512];
  while (fgets (buf, sizeof (buf), file))
    {

      long long vocab_id = -1;
      if (sscanf (buf, "%lld", &vocab_id) != 1)
        goto on_error;

      if (vocab_buf_size <= vocab_id)
        {
          void *rbuf
              = realloc (tokenizer->token2text,
                         sizeof (char *) * (vocab_buf_size + vocab_buf_chunk));
          if (rbuf == NULL)
            goto on_error;
          memset (rbuf + vocab_buf_size, 0, sizeof (char *) * vocab_buf_chunk);
          tokenizer->token2text = rbuf;
          vocab_buf_size += vocab_buf_chunk;
        }

      char *text_len_str = strrchr (buf, ' ');
      if (text_len_str == NULL)
        goto on_error;

      size_t text_len = 0;
      if (sscanf (text_len_str, "%zd", &text_len) != 1)
        goto on_error;

      char *text_start = strchr (buf, ' ');
      if (text_start == NULL)
        goto on_error;
      text_start += 1;
      size_t text_repr_size = (size_t)(text_len_str - text_start);
      char text_repr[text_repr_size + 1];
      strncpy (text_repr, text_start, text_repr_size);

      size_t py_str_len = 0;
      char *text = python_str (text_repr, &py_str_len);
      if (text == NULL)
        goto on_error;

      if (py_str_len != text_len)
        {
          free (text);
          goto on_error;
        }

      struct RWKVTokenizerEntry *entry = &tokenizer->root;
      for (size_t k = 0; k < strlen (text); k++)
        {
          if (entry->next[(unsigned char)text[k]] != NULL)
            {
              entry = entry->next[(unsigned char)text[k]];
              continue;
            }

          entry->next[(unsigned char)text[k]]
              = malloc (sizeof (struct RWKVTokenizerEntry));
          if (entry->next[(unsigned char)text[k]] == NULL)
            {
              free (text);
              goto on_error;
            }
          memset (entry->next[(unsigned char)text[k]], 0,
                  sizeof (struct RWKVTokenizerEntry));

          entry->next[(unsigned char)text[k]]->prev = entry;
          entry = entry->next[(unsigned char)text[k]];
          entry->id = -1;
          entry->ch = text[k];
        }

      entry->id = vocab_id;
      tokenizer->token2text[vocab_id] = text;

      if (vocab_size < vocab_id)
        vocab_size = vocab_id;
    }

  tokenizer->vocab_size = vocab_size + 1;

  fclose (file);
  return true;
on_error:
  fclose (file);
  return false;
}

static void
rwkv_tokenizer_free (struct RWKVTokenizer *tokenizer)
{
  struct RWKVTokenizer empty_tokenizer = { 0 };
  if (!memcmp (tokenizer, &empty_tokenizer, sizeof (struct RWKVTokenizer)))
    return;

  size_t idx = -1;
  for (struct RWKVTokenizerEntry *entry = &tokenizer->root; entry != NULL;)
    {
      size_t search_idx = -1;
      for (size_t k = 0; k < 256; k++)
        {
          if (entry->next[k] == NULL)
            continue;
          entry = entry->next[k];
          idx = k;
          search_idx = k;
          break;
        }

      if ((search_idx == -1) && (idx != -1))
        {
          struct RWKVTokenizerEntry *_entry = entry->prev;
          if (entry == &tokenizer->root)
            break;
          _entry->next[idx] = NULL;
          free (entry);
          entry = _entry;
          idx = -1;
        }
      else if ((search_idx == -1) && (idx == -1))
        entry = entry->prev;
    }

  for (size_t k = 0; k < tokenizer->vocab_size; k++)
    {
      if (tokenizer->token2text[k] != NULL)
        free (tokenizer->token2text[k]);
    }
  free (tokenizer->token2text);
}

static int64_t *
rwkv_tokenizer_tokenize (struct RWKVTokenizer *tokenizer, char *text,
                         size_t *tokens_len)
{
  int64_t *tokens;
  size_t _tokens_len = 0;
  size_t tokens_size = 512;
  size_t tokens_size_chunk = 512;

  tokens = malloc (sizeof (int64_t) * tokens_size_chunk);
  if (tokens == NULL)
    goto on_error;

  struct RWKVTokenizerEntry *entry = &tokenizer->root;
  for (size_t k = 0; k < strlen (text); k++)
    {
      if (entry->next[(unsigned char)text[k]] != NULL)
        {
          entry = entry->next[(unsigned char)text[k]];
          continue;
        }

      while (entry->id == -1)
        {
          entry = entry->prev;
          k--;
        }

      if (_tokens_len == tokens_size)
        {
          void *buf = realloc (
              tokens, sizeof (int64_t) * (_tokens_len + tokens_size_chunk));
          if (buf == NULL)
            goto on_error;
          tokens = buf;
        }

      tokens[_tokens_len++] = entry->id;
      entry = &tokenizer->root;
      k--;
    }

  if (entry->id != -1)
    {
      if (_tokens_len == tokens_size)
        {
          void *buf = realloc (tokens, sizeof (int64_t) * (_tokens_len + 1));
          if (buf == NULL)
            goto on_error;
          tokens = buf;
        }

      tokens[_tokens_len++] = entry->id;
    }

  *tokens_len = _tokens_len;
  return tokens;

on_error:
  free (tokens);
  return NULL;
}

static char *
rwkv_tokenizer_detokenize (struct RWKVTokenizer *tokenizer, int64_t *tokens,
                           size_t tokens_len)
{
  const size_t text_size_chunk = tokens_len * 8;
  size_t text_size = text_size_chunk;
  size_t text_len = 0;
  char *text = malloc (text_size);
  if (text == NULL)
    return NULL;

  for (size_t k = 0; k < tokens_len; k++)
    {
      const char *token_text = tokenizer->token2text[tokens[k]];
      if (token_text == NULL)
        token_text = " ";
      size_t token_text_len = strlen (token_text);
      if ((text_size - text_len) <= token_text_len)
        {
          void *buf = realloc (text, text_size + text_size_chunk);
          if (buf == NULL)
            goto on_error;
          text = buf;
          text_size += text_size_chunk;
        }

      info_printf ("text_size: %zd, text_len: %zd, token_text_len: %zd\n",
                   text_size, text_len, token_text_len);
      assert (token_text_len == strlen (token_text));
      strcpy (text + text_len, token_text);
      text_len += token_text_len;
    }

  return text;

on_error:
  free (text);
  return NULL;
}

static bool
retrieve_dim (OrtTypeInfo *info, size_t *ndim, int64_t **dim,
              const char ***axis_name, OrtStatus **status)
{
  assert (info != NULL);
  assert (ndim != NULL);
  assert (axis_name != NULL);

  size_t _ndim = 0;
  int64_t *_dim = NULL;
  const char **_axis_name = NULL;

  const OrtTensorTypeAndShapeInfo *tensor_info;
  OrtStatus *_status
      = g_ort_api->CastTypeInfoToTensorInfo (info, &tensor_info);
  if (_status)
    goto on_ort_error;

  _status = g_ort_api->GetDimensionsCount (tensor_info, &_ndim);
  if (_status)
    goto on_ort_error;

  _dim = malloc (sizeof (int64_t) * _ndim);
  if (_dim == NULL)
    goto finalize;

  _axis_name = malloc (sizeof (char *) * _ndim);
  if (_axis_name == NULL)
    goto finalize;

  _status = g_ort_api->GetDimensions (tensor_info, _dim, _ndim);
  if (_status)
    goto on_ort_error;

  _status = g_ort_api->GetSymbolicDimensions (tensor_info, _axis_name, _ndim);
  if (_status)
    goto on_ort_error;

  *ndim = _ndim;
  *dim = _dim;
  *axis_name = _axis_name;

  return true;
on_ort_error:
  if (status)
    *status = _status;
  else
    g_ort_api->ReleaseStatus (_status);
finalize:
  if (_axis_name)
    free (_axis_name);
  if (_dim)
    free (_dim);
  return false;
}

static bool
rwkv_init (struct RWKV *rwkv, OrtStatus **status)
{
  assert (rwkv != NULL);

  OrtStatus *_status = NULL;

  memset (rwkv, 0, sizeof (struct RWKV));
  info_printf ("Creating OrtSessionOptions.\n");
  _status = g_ort_api->CreateSessionOptions (&rwkv->session_options);
  if (_status)
    goto on_ort_error;

  info_printf ("Creating OrtSession.\n");
  _status = g_ort_api->CreateSession (g_ort_env, g_args.model,
                                      rwkv->session_options, &rwkv->session);
  if (_status)
    goto on_ort_error;

  _status = g_ort_api->CreateRunOptions (&rwkv->run_options);
  if (_status)
    goto on_ort_error;

  info_printf ("Getting input count.\n");
  _status = g_ort_api->SessionGetInputCount (rwkv->session, &rwkv->ninput);
  if (_status)
    goto on_ort_error;

  info_printf ("Getting output count.\n");
  _status = g_ort_api->SessionGetOutputCount (rwkv->session, &rwkv->noutput);
  if (_status)
    goto on_ort_error;

  rwkv->input_name = malloc (sizeof (char *) * rwkv->ninput);
  if (rwkv->input_name == NULL)
    goto finalize;
  memset (rwkv->input_name, 0, sizeof (char *) * rwkv->ninput);

  rwkv->output_name = malloc (sizeof (char *) * rwkv->noutput);
  if (rwkv->output_name == NULL)
    goto finalize;
  memset (rwkv->output_name, 0, sizeof (char *) * rwkv->noutput);

  size_t nio = rwkv->ninput + rwkv->noutput;
  rwkv->ndim = malloc (sizeof (size_t) * nio);
  if (rwkv->ndim == NULL)
    goto finalize;
  rwkv->dim = malloc (sizeof (int64_t *) * nio);
  if (rwkv->dim == NULL)
    goto finalize;
  memset (rwkv->dim, 0, sizeof (int64_t *) * nio);
  rwkv->axis_name = malloc (sizeof (char **) * nio);
  if (rwkv->axis_name == NULL)
    goto finalize;

  for (size_t i = 0; i < rwkv->ninput; i++)
    {
      OrtTypeInfo *type_info;

      _status = g_ort_api->SessionGetInputName (
          rwkv->session, i, g_ort_allocator, &rwkv->input_name[i]);
      if (_status)
        goto on_ort_error;

      _status
          = g_ort_api->SessionGetInputTypeInfo (rwkv->session, i, &type_info);
      if (_status)
        goto on_ort_error;

      if (!retrieve_dim (type_info, &rwkv->ndim[i], &rwkv->dim[i],
                         &rwkv->axis_name[i], status))
        {
          g_ort_api->ReleaseTypeInfo (type_info);
          goto finalize;
        }
      g_ort_api->ReleaseTypeInfo (type_info);
    }

  for (size_t o = 0; o < rwkv->noutput; o++)
    {
      OrtTypeInfo *type_info;

      _status = g_ort_api->SessionGetOutputName (
          rwkv->session, o, g_ort_allocator, &rwkv->output_name[o]);
      if (_status)
        goto on_ort_error;

      _status
          = g_ort_api->SessionGetOutputTypeInfo (rwkv->session, o, &type_info);
      if (_status)
        goto on_ort_error;

      if (!retrieve_dim (type_info, &rwkv->ndim[rwkv->ninput + o],
                         &rwkv->dim[rwkv->ninput + o],
                         &rwkv->axis_name[rwkv->ninput + o], status))
        {
          g_ort_api->ReleaseTypeInfo (type_info);
          goto finalize;
        }

      g_ort_api->ReleaseTypeInfo (type_info);
    }

  return true;
on_ort_error:
  if (status)
    *status = _status;
  else
    g_ort_api->ReleaseStatus (_status);
finalize:
  if (rwkv->axis_name)
    {
      for (size_t a = 0; a < nio; a++)
        free (rwkv->axis_name[a]);
      free (rwkv->axis_name);
    }
  if (rwkv->dim)
    {
      for (size_t i = 0; i < nio; i++)
        {
          if (rwkv->dim[i] != NULL)
            free (rwkv->dim[i]);
        }
      free (rwkv->dim);
    }
  if (rwkv->ndim)
    free (rwkv->ndim);
  if (rwkv->output_name)
    {
      for (size_t o = 0; o < rwkv->noutput; o++)
        if (rwkv->output_name[o])
          g_ort_allocator->Free (g_ort_allocator, rwkv->output_name[o]);
      free (rwkv->output_name);
    }
  if (rwkv->input_name)
    {
      for (size_t i = 0; i < rwkv->ninput; i++)
        if (rwkv->input_name[i])
          g_ort_allocator->Free (g_ort_allocator, rwkv->input_name[i]);
      free (rwkv->input_name);
    }
  if (rwkv->run_options)
    g_ort_api->ReleaseRunOptions (rwkv->run_options);
  if (rwkv->session)
    g_ort_api->ReleaseSession (rwkv->session);
  if (rwkv->session_options)
    g_ort_api->ReleaseSessionOptions (rwkv->session_options);
  return false;
}

static void
rwkv_free (struct RWKV *rwkv)
{
  struct RWKV empty_rwkv = { 0 };

  if (!memcmp (rwkv, &empty_rwkv, sizeof (struct RWKV)))
    return;

  size_t nio = rwkv->ninput + rwkv->noutput;

  for (size_t k = 0; k < nio; k++)
    free (rwkv->axis_name[k]);
  free (rwkv->axis_name);
  for (size_t i = 0; i < nio; i++)
    free (rwkv->dim[i]);
  free (rwkv->dim);
  free (rwkv->ndim);
  for (size_t o = 0; o < rwkv->noutput; o++)
    {
      info_printf ("Freeing output_name[%zd]: %s\n", o, rwkv->output_name[o]);
      g_ort_allocator->Free (g_ort_allocator, rwkv->output_name[o]);
    }
  free (rwkv->output_name);
  for (size_t i = 0; i < rwkv->ninput; i++)
    {
      info_printf ("Freeing output_name[%zd]: %s\n", i, rwkv->input_name[i]);
      g_ort_allocator->Free (g_ort_allocator, rwkv->input_name[i]);
    }
  free (rwkv->input_name);
  g_ort_api->ReleaseRunOptions (rwkv->run_options);
  g_ort_api->ReleaseSession (rwkv->session);
  g_ort_api->ReleaseSessionOptions (rwkv->session_options);
}

static bool
rwkv_init_io (struct RWKV *rwkv, OrtValue ***inputs, OrtValue ***outputs,
              OrtStatus **status)
{
  assert (rwkv != NULL);
  assert (inputs != NULL);
  assert (outputs != NULL);
  assert (rwkv->ninput == rwkv->noutput);

  OrtStatus *_status = NULL;
  OrtValue **rwkv_input = NULL;
  OrtValue **rwkv_output = NULL;

  rwkv_input = malloc (sizeof (OrtValue *) * rwkv->ninput);
  if (rwkv_input == NULL)
    goto on_error;
  memset (rwkv_input, 0, sizeof (OrtValue *) * rwkv->ninput);

  for (size_t i = 1; i < rwkv->ninput; i++)
    {
      int64_t dim[rwkv->ndim[i]];
      memcpy (dim, rwkv->dim[i], sizeof (int64_t) * rwkv->ndim[i]);
      dim[0] = 1;
      _status = g_ort_api->CreateTensorAsOrtValue (
          g_ort_allocator, dim, rwkv->ndim[i],
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &rwkv_input[i]);
      if (_status)
        goto on_ort_error;

      float *data = NULL;
      _status
          = g_ort_api->GetTensorMutableData (rwkv_input[i], (void **)&data);
      if (_status)
        goto on_ort_error;

      size_t e = 1;
      for (size_t d = 0; d < rwkv->ndim[i]; d++)
        e *= rwkv->dim[i][d] == -1 ? 1 : rwkv->dim[i][d];
      memset (data, 0, e * sizeof (float));
    }

  rwkv_output = malloc (sizeof (OrtValue *) * rwkv->noutput);
  if (rwkv_output == NULL)
    goto on_error;
  rwkv_output[0] = NULL;
  memcpy (rwkv_output + 1, rwkv_input + 1,
          sizeof (OrtValue *) * (rwkv->noutput - 1));

  *inputs = rwkv_input;
  *outputs = rwkv_output;

  return true;
on_ort_error:
  if (status)
    *status = _status;
  else
    g_ort_api->ReleaseStatus (_status);
on_error:
  if (rwkv_input != NULL)
    {
      for (size_t i = 1; i < rwkv->ninput; i++)
        if (rwkv_input[i] != NULL)
          g_ort_api->ReleaseValue (rwkv_input[i]);
      free (rwkv_input);
    }
  if (rwkv_output != NULL)
    free (rwkv_output);
  return false;
}

static void
rwkv_free_io (struct RWKV *rwkv, OrtValue **inputs, OrtValue **outputs)
{
  for (size_t i = 1; i < rwkv->ninput; i++)
    g_ort_api->ReleaseValue (inputs[i]);
  free (inputs);

  if (outputs[0] != NULL)
    free (outputs[0]);
  free (outputs);
}

static OrtStatus *
rwkv_run (struct RWKV *rwkv, const OrtValue *const *input, OrtValue **output)
{
  return g_ort_api->Run (
      rwkv->session, rwkv->run_options, (const char **)rwkv->input_name, input,
      rwkv->ninput, (const char **)rwkv->output_name, rwkv->noutput, output);
}

static int64_t
rwkv_decode (struct RWKV *rwkv, OrtValue *output, OrtStatus **status)
{
  OrtTensorTypeAndShapeInfo *tensor_info = NULL;
  OrtStatus *_status = g_ort_api->GetTensorTypeAndShape (output, &tensor_info);
  if (_status)
    goto on_ort_error;

  size_t ndim;
  _status = g_ort_api->GetDimensionsCount (tensor_info, &ndim);
  if (_status)
    goto on_ort_error;
  if (ndim != 3)
    {
      error_printf ("Logit's ndim should 3.\n");
      abort ();
    }

  int64_t dim[3] = { 0 };
  _status = g_ort_api->GetDimensions (tensor_info, dim, ndim);
  if (_status)
    goto on_ort_error;
  if (dim[0] != 1)
    {
      error_printf ("Unsupported batch size.\n");
      abort ();
    }

  float *data;
  _status = g_ort_api->GetTensorMutableData (output, (void **)&data);
  if (_status)
    goto on_ort_error;
  data += dim[0] * (dim[1] - 1) * dim[2];

  assert (rwkv->ndim[rwkv->ninput] == 3);
  int64_t vocab_size = dim[2];

  float T = 0.7;

  double sum = 0;
  for (int64_t k = 0; k < vocab_size; k++)
    {
      data[k] = exp (data[k] / T);
      sum += data[k];
    }

  for (int64_t k = 0; k < vocab_size; k++)
    data[k] /= sum;

  float r = (float)rand () / (float)RAND_MAX;

  double s = 0;
  int64_t k = 0;
  for (; k < vocab_size; k++)
    {
      s += data[k];
      if (r < s)
        goto on_return;
    }

on_return:
  g_ort_api->ReleaseTensorTypeAndShapeInfo (tensor_info);
  return k;

on_ort_error:
  if (status)
    *status = _status;
  else
    g_ort_api->ReleaseStatus (_status);

  g_ort_api->ReleaseTensorTypeAndShapeInfo (tensor_info);
  return -1;
}

int
main (int argc, char **argv)
{
  struct RWKV rwkv = { 0 };
  struct RWKVTokenizer tokenizer = { 0 };
  OrtValue **rwkv_input = NULL;
  OrtValue **rwkv_output = NULL;
  OrtValue *decode_input = NULL;
  char *buf = NULL;
  char *template_buf = NULL;

  if (!parse_args (argc, argv))
    {
      error_printf (
          "Given arguments are invalid. Try -h or --help to see usage.\n");
      return 1;
    }

  if (g_args.help || !is_valid_args (&g_args))
    {
      help ();
      return 0;
    }

  info_printf ("Starting program.\n");

  g_ort_api = OrtGetApiBase ()->GetApi (ORT_API_VERSION);
  assert (g_ort_api != NULL);

  OrtStatus *status = NULL;
  if (g_args.verbose)
    {
      info_printf ("Creating OrtEnv with log level verbose.\n");
      status = g_ort_api->CreateEnv (ORT_LOGGING_LEVEL_VERBOSE, "ONNXRuntime",
                                     &g_ort_env);
    }
  else
    {
      info_printf ("Creating OrtEnv with log level warning.\n");
      status = g_ort_api->CreateEnv (ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime",
                                     &g_ort_env);
    }

  if (status)
    goto on_ort_error;

  info_printf ("Getting OrtAllocator with default options.\n");
  status = g_ort_api->GetAllocatorWithDefaultOptions (&g_ort_allocator);
  if (status)
    goto on_ort_error;

  if (!rwkv_init (&rwkv, &status))
    goto on_ort_error;

  if (!rwkv_tokenizer_init (&tokenizer, g_args.tokenizer))
    {
      error_printf ("Failed to initialize tokenizer.\n");
      goto on_error;
    }

  if (!rwkv_init_io (&rwkv, &rwkv_input, &rwkv_output, &status))
    {
      if (status)
        goto on_ort_error;
      goto on_malloc_error;
    }

  status = g_ort_api->CreateTensorAsOrtValue (
      g_ort_allocator, (int64_t[]){ 1, 1 }, 2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &decode_input);
  if (status)
    goto on_ort_error;
  int64_t *decode_input_data = NULL;
  status = g_ort_api->GetTensorMutableData (decode_input,
                                            (void **)&decode_input_data);
  if (status)
    goto on_ort_error;

  size_t buf_size = 4096;
  buf = malloc (buf_size);
  if (buf == NULL)
    goto on_malloc_error;
  size_t template_buf_size = 4096;
  template_buf = malloc (template_buf_size);
  if (template_buf == NULL)
    goto on_malloc_error;

  assert (rwkv.ninput == rwkv.noutput);
  while (fgets (buf, buf_size, stdin))
    {
      size_t buf_len = strlen (buf);

      while (buf_len == (buf_size - 1))
        {
          void *_buf = realloc (buf, buf_size + 4096);
          if (_buf == NULL)
            goto on_malloc_error;
          buf = _buf;
          if (fgets (buf + buf_size, 4096, stdin) == NULL)
            break;
          buf_size += 4096;
        }

      for (size_t l = strlen (buf) - 1; l != 0; l--)
        if (iscntrl (buf[l]) || isspace (buf[l]))
          buf[l] = 0;
        else
          break;

      if (!strcmp (buf, "!exit"))
        break;

      const char chat_template[] = "User: %s\n\nAssistant:";
      size_t template_buf_need = strlen (chat_template) + strlen (buf) + 1;
      if (buf_size <= template_buf_need)
        {
          void *_buf = realloc (template_buf, template_buf_need);
          if (_buf == NULL)
            goto on_malloc_error;
          template_buf = _buf;
          template_buf_size = template_buf_need;
        }

      sprintf (template_buf, chat_template, buf);

      size_t tokens_len;
      int64_t *tokens
          = rwkv_tokenizer_tokenize (&tokenizer, template_buf, &tokens_len);
      if (tokens == NULL)
        {
          error_printf ("Failed to tokenize.\n");
          goto on_error;
        }

      info_printf ("Input token-length: %zd\n", tokens_len);

      OrtValue *input = NULL;
      status = g_ort_api->CreateTensorAsOrtValue (
          g_ort_allocator, (int64_t[]){ 1, tokens_len }, 2,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input);
      if (status)
        {
          free (tokens);
          goto on_ort_error;
        }
      int64_t *data = NULL;
      status = g_ort_api->GetTensorMutableData (input, (void **)&data);
      if (status)
        {
          free (tokens);
          goto on_ort_error;
        }
      memcpy (data, tokens, sizeof (int64_t) * tokens_len);
      free (tokens);

      rwkv_input[0] = input;

      status = rwkv_run (&rwkv, (const OrtValue **)rwkv_input, rwkv_output);
      if (status)
        goto on_ort_error;
      memcpy (rwkv_input + 1, rwkv_output + 1,
              sizeof (OrtValue *) * (rwkv.ninput - 1));

      g_ort_api->ReleaseValue (input);
      rwkv_input[0] = decode_input;

      do
        {
          int64_t output_token = rwkv_decode (&rwkv, rwkv_output[0], &status);
          if (status)
            goto on_ort_error;

          g_ort_api->ReleaseValue (rwkv_output[0]);
          rwkv_output[0] = NULL;

          char *text
              = rwkv_tokenizer_detokenize (&tokenizer, &output_token, 1);
          if (text == NULL)
            {
              error_printf ("Decoding token to text failed.\n");
              goto on_error;
            }

          printf ("%s", text);

          if (strstr (text, "\n\n") != NULL)
            {
              g_ort_api->ReleaseValue (rwkv_output[0]);
              rwkv_output[0] = NULL;
              free (text);
              fflush (stdout);
              break;
            }

          free (text);

          *decode_input_data = output_token;
          status
              = rwkv_run (&rwkv, (const OrtValue **)rwkv_input, rwkv_output);
          if (status)
            goto on_ort_error;
        }
      while (true);
    }

  free (template_buf);
  free (buf);
  g_ort_api->ReleaseValue (decode_input);
  rwkv_free_io (&rwkv, rwkv_input, rwkv_output);
  rwkv_free (&rwkv);
  rwkv_tokenizer_free (&tokenizer);
  g_ort_api->ReleaseEnv (g_ort_env);

  return 0;

on_ort_error:;
  error_printf ("%s\n", g_ort_api->GetErrorMessage (status));
  g_ort_api->ReleaseStatus (status);
  goto finalize;
on_malloc_error:;
  error_printf ("Failed to allocate memory.\n");
  goto finalize;
on_error:;

finalize:;
  if (template_buf != NULL)
    free (template_buf);
  if (buf != NULL)
    free (buf);
  if (decode_input != NULL)
    g_ort_api->ReleaseValue (decode_input);
  if ((rwkv_input != NULL) && (rwkv_output != NULL))
    rwkv_free_io (&rwkv, rwkv_input, rwkv_output);
  rwkv_free (&rwkv);
  rwkv_tokenizer_free (&tokenizer);
  g_ort_api->ReleaseEnv (g_ort_env);
  return 1;
}
