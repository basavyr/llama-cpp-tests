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

ContextInit initialize_context(llama_model *model, int n_input_tokens, const int max_output_tokens)
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

int main()
{
    std::string title = "Testing inference using `llama.cpp`";
    printf("%s\n", title.c_str());

    const std::string input_prompt = "What is quantum computing?";
    const int MAX_OUT_TOKENS = 128;

    ggml_backend_load_all();
    const std::string MODEL_PATH = "/Users/svc_sps/.lmstudio/models/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf";

    ModelParams model_config = ModelParams();
    model_config.n_gpu_layers = 99;
    model_config.model_path = MODEL_PATH;
    model_config.model_params = llama_model_default_params();

    llama_model *model = llama_model_load_from_file(model_config.model_path.c_str(), model_config.model_params);

    if (model == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);

    // get number of tokens of the input prompt
    /*
    args:
    const struct llama_vocab * vocab,
                  const char * text,
                     int32_t   text_len,
                 llama_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special)
    */
    const int n_input_tokens = -llama_tokenize(vocab, input_prompt.c_str(), input_prompt.size(), NULL, 0, true, true);
    // tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_input_tokens);
    int n_tokens_max = prompt_tokens.size();

    if (llama_tokenize(vocab, input_prompt.c_str(), input_prompt.size(), prompt_tokens.data(), n_tokens_max, true, true) < 0)
    {
        fprintf(stderr, "failed to tokenize the prompt\n");
    }

    fprintf(stdout, "n.o. prompt tokens: %d\nn_tokens_max: %d\n", n_input_tokens, n_tokens_max);

    // check the type of the prompt tokens
    // std::cout << type_name<decltype(prompt_tokens)>();
    std::cout << input_prompt << " -> ";
    for (auto &&n : prompt_tokens)
    {
        std::cout << n << " ";
    }
    std::cout << std::endl;

    ContextInit context = initialize_context(model, n_input_tokens, MAX_OUT_TOKENS);

    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    llama_sampler *sampler = llama_sampler_chain_init(sampler_params);

    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    for (auto &&t : prompt_tokens)
    {
        print_token(t, vocab);
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
    if (decoder_start_token_id == LLAMA_TOKEN_NULL)
    {
        decoder_start_token_id = llama_vocab_bos(vocab);
    }
    batch = llama_batch_get_one(&decoder_start_token_id, 1);

    print_token(decoder_start_token_id, vocab);

    return 0;
}