#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>

#include <string_view>

// source: https://stackoverflow.com/questions/81870/is-it-possible-to-print-the-name-of-a-variables-type-in-standard-c
template <typename T>
constexpr auto type_name()
{
    std::string_view name, prefix, suffix;
#ifdef __clang__
    name = __PRETTY_FUNCTION__;
    prefix = "auto type_name() [T = ";
    suffix = "]";
#elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
    prefix = "constexpr auto type_name() [with T = ";
    suffix = "]";
#elif defined(_MSC_VER)
    name = __FUNCSIG__;
    prefix = "auto __cdecl type_name<";
    suffix = ">(void)";
#endif
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return name;
}

struct ModelParams
{
    int n_gpu_layers;
    std::string model_path;
    llama_model_params model_params;
};

struct ContextInit
{
    llama_context_params params;
    llama_context *ctx;
};

ContextInit get_context(llama_model *model, int n_input_tokens, int max_output_tokens)
{
    // llama model should be passed as a pointer, since it is an opaque pointer and it cannot be used via &
    // source: https://stackoverflow.com/questions/7058339/when-should-i-use-pointers-instead-of-references-in-api-design/7058373#7058373
    llama_context_params params = llama_context_default_params();
    params.n_ctx = n_input_tokens + max_output_tokens - 1;
    params.n_batch = n_input_tokens;
    params.no_perf = false;

    llama_context *ctx = llama_init_from_model(model, params);

    return ContextInit{params, ctx};
}

void print_token(int token_id, const llama_vocab *vocab)
{
    char buffer[128];
    int token_to_piece = llama_token_to_piece(vocab, token_id, buffer, sizeof(buffer), 0, true);

    std::string token_as_str(buffer, token_to_piece);
    printf("Token: %i -> Str: %s\n", token_id, token_as_str.c_str());
}

llama_sampler *get_sampler()
{
    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    llama_sampler *sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    return sampler;
}

struct llama_model_and_vocab
{
    llama_model *model;
    const llama_vocab *vocab;
};

llama_model_and_vocab get_model_and_vocab(const std::string &model_path)
{
    ModelParams model_config = ModelParams();
    model_config.model_path = model_path;
    model_config.n_gpu_layers = 999;
    model_config.model_params = llama_model_default_params();

    // model and vocab
    llama_model *model = llama_model_load_from_file(model_config.model_path.c_str(), model_config.model_params);
    const llama_vocab *vocab = llama_model_get_vocab(model);

    if (model == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        exit(EXIT_FAILURE);
    }

    return llama_model_and_vocab{model, vocab};
}

struct llama_tokenizer
{
    const int n_input_tokens;
    std::vector<llama_token> prompt_tokens;
    ContextInit context;
};

llama_tokenizer tokenize_prompt(llama_model_and_vocab *model_and_vocab, const std::string &input_prompt, const int max_output_tokens)
{
    /*
    llama_tokenize args:
    const struct llama_vocab * vocab,
                const char * text,
                    int32_t   text_len,
                llama_token * tokens,
                    int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special)
    */
    const int n_input_tokens = -llama_tokenize(model_and_vocab->vocab, input_prompt.c_str(), input_prompt.size(), NULL, 0, true, true);
    ContextInit context = get_context(model_and_vocab->model, n_input_tokens, max_output_tokens);
    // tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_input_tokens);
    int n_tokens_max = prompt_tokens.size();

    if (llama_tokenize(model_and_vocab->vocab, input_prompt.c_str(), input_prompt.size(), prompt_tokens.data(), n_tokens_max, true, true) < 0)
    {
        fprintf(stderr, "failed to tokenize the prompt\n");
        exit(EXIT_FAILURE);
    }

    return llama_tokenizer{n_input_tokens, prompt_tokens, context};
}

void batch_decode(
    llama_model *model,
    const llama_vocab *vocab,
    std::vector<llama_token> *prompt_tokens,
    ContextInit *context,
    int n_input_tokens,
    const int max_output_tokens)
{
    /*
    llama_batch args:
        int32_t         n_tokens;
        llama_token  *  token;
        float        *  embd;
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits;
    */
    llama_batch batch = llama_batch_get_one(prompt_tokens->data(), prompt_tokens->size()); // use `->` instead of `.` when the object is passed as a pointer

    // for decoder it should be -1
    llama_token decoder_start_token_id = llama_model_decoder_start_token(model);

    if (decoder_start_token_id == LLAMA_TOKEN_NULL)
    {
        decoder_start_token_id = llama_vocab_bos(vocab);
    }

    llama_sampler *sampler = get_sampler();

    int decode_steps = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_input_tokens + max_output_tokens;)
    {
        llama_decode(context->ctx, batch); // use `->` instead of `.` when the object is passed as a pointer
        n_pos += batch.n_tokens;

        // sampling the next token
        {
            new_token_id = llama_sampler_sample(sampler, context->ctx, -1);
            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id))
            {
                break;
            }

            char buffer[128];
            int current_token_id = llama_token_to_piece(vocab, new_token_id, buffer, sizeof(buffer), 0, true); // this must be positive
            std::string token_string(buffer, current_token_id);
            printf("%s", token_string.c_str());
            fflush(stdout);

            batch = llama_batch_get_one(&new_token_id, 1);

            decode_steps += 1;
        }
    }
    printf("\n\nFinished batch decoding (%i steps)\n", decode_steps);
}

int main()
{
    const std::string input_prompt = "What is quantum computing?";
    const int max_output_tokens = 128;

    ggml_backend_load_all();

    const std::string model_path = "/Users/svc_sps/.lmstudio/models/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf";
    llama_model_and_vocab model_and_vocab = get_model_and_vocab(model_path);

    llama_tokenizer tokenizer = tokenize_prompt(&model_and_vocab, input_prompt, max_output_tokens);
    batch_decode(model_and_vocab.model, model_and_vocab.vocab, &tokenizer.prompt_tokens, &tokenizer.context, tokenizer.n_input_tokens, max_output_tokens);

    return 0;
}