from functools import partial
import global_vars

import gradio as gr

from gens.batch_gen import get_output_batch
from miscs.strings import SPECIAL_STRS, PLEASE_SPECIFY, SORRY_MESAGE
from miscs.constants import num_of_characters_to_keep
from miscs.utils import common_post_process, post_processes_batch, post_process_stream

from chats.prompts import generate_prompt
from hyde_utils import suggest_documents, filter_hyde_context, convert_dict2context

from fuzzywuzzy import fuzz

from const_app import APPS_DICT

from collections import defaultdict


def chat_stream(
    context,
    instruction,
    state_chatbot,
    state_recieved_docs,
    encoder=None,
    index=None,
    dataset=None
):
    if global_vars.constraints_config.len_exceed(context, instruction):
        raise gr.Error("context or prompt is too long!")
    
    gen_prompt = partial(generate_prompt, ctx_indicator="### Input:", user_indicator="### Instruction:", ai_indicator="### Response:")
    bot_summarized_response = ''
    app_context = ""
    no_info = False
    no_recent_retrieve = False
    if len(state_recieved_docs) == 0:
        scores = []
        for k in APPS_DICT:
            score = fuzz.partial_ratio(k, instruction)
            scores.append((k, score)) 
        
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        filtered_dict = defaultdict(list)
        for k, score in scores[:20]:
            filtered_dict[k] = APPS_DICT[k]
    
        app_context = convert_dict2context(filtered_dict)
    
    filtered_context_docs = filter_hyde_context(instruction, state_recieved_docs, encoder)
    suggested_documents = None
    context_suggested = ""
    context_with_documents = ""
    context_short = ""
    # context_short_descs = "\n".join([
    #         dataset[ind]['short_desc'] for ind, embed in state_recieved_docs[:2]
    # ])
    # context_with_documents = context + '\n' + app_context +"\n" + context_short_descs
    if len(filtered_context_docs) == 0:
        suggested_documents = suggest_documents(instruction, encoder, index, dataset)
        state_recieved_docs += [
            (ind, embed) for ind, text, embed in suggested_documents
        ]
        suggestet_docs_texts = [
            text for ind, text, embed in suggested_documents
        ]
        context_suggested = "\n".join(suggestet_docs_texts)
    else:
        context_texts = [
            dataset[ind]['text'] for ind, embed in filtered_context_docs
        ]
        #context_with_documents += "\n".join(context_texts)
    #if len(state_recieved_docs) > 10:
    if len(state_recieved_docs) == 0 and len(suggested_documents) == 0:
        no_info = True
    if len(state_recieved_docs) == 0:
        no_recent_retrieve = True
    while len(state_recieved_docs) > 5: state_recieved_docs.pop(0)
    
    context_short_descs = "\n".join([
            dataset[ind]['short_desc'] for ind, embed in state_recieved_docs[:3]
    ])
    context_long_descs = "\n".join([
            dataset[ind]['text'] for ind, embed in state_recieved_docs[3:]
    ])

    context_with_documents = context + "\n" + context_short_descs + "\n" + context_long_descs + '\n'
    

    # user input should be appropriately formatted (don't be confused by the function name)
    instruction_display = common_post_process(instruction)
    instruction_prompt, conv_length = gen_prompt(instruction, state_chatbot, context_with_documents)
    #print("INSTRUCTION PROMPT: {}".format(instruction_prompt))
    
    if global_vars.constraints_config.conv_len_exceed(conv_length):
        instruction_prompt = gen_prompt(SPECIAL_STRS["summarize"], state_chatbot, context_with_documents, partial=True)[0]
        
        state_chatbot = state_chatbot + [
            (
                None, 
                "![](https://s2.gifyu.com/images/icons8-loading-circle.gif) too long conversations, so let's summarize..."
            )
        ]
        yield (state_chatbot, state_chatbot, context, state_recieved_docs)
        
        bot_summarized_response = get_output_batch(
            global_vars.model, global_vars.tokenizer, [instruction_prompt], global_vars.gen_config_summarization
        )[0]
        bot_summarized_response = bot_summarized_response.split("### Response:")[-1].strip()
        
        state_chatbot[-1] = (
            None, 
            "✅ summarization is done and set as context"
        )
        print(f"bot_summarized_response: {bot_summarized_response}")
        yield (state_chatbot, state_chatbot, f"{context}. {bot_summarized_response}".strip(), state_recieved_docs)
        
    instruction_prompt = gen_prompt(instruction, state_chatbot, f"{context_with_documents} {bot_summarized_response}")[0]
    print(instruction_prompt)
    if no_info:
        bot_response = PLEASE_SPECIFY
    else:
        bot_response = global_vars.stream_model(
            instruction_prompt,
            max_tokens=256,
            temperature=0.75,
            top_p=0.80
        )
    
    instruction_display = None if instruction_display == SPECIAL_STRS["continue"] else instruction_display
    state_chatbot = state_chatbot + [(instruction_display, None)]
    yield (state_chatbot, state_chatbot, f"{context}. {bot_summarized_response}".strip(), state_recieved_docs)
    
    prev_index = 0
    agg_tokens = ""
    cutoff_idx = 0
    for tokens in bot_response:
        tokens = tokens.strip()
        cur_token = tokens[prev_index:]
        
        if "#" in cur_token and agg_tokens == "":
            cutoff_idx = tokens.find("#")
            agg_tokens = tokens[cutoff_idx:]

        if agg_tokens != "":
            if len(agg_tokens) < len("### Instruction:") :
                agg_tokens = agg_tokens + cur_token
            elif len(agg_tokens) >= len("### Instruction:"):
                if tokens.find("### Instruction:") > -1:
                    processed_response, _ = post_process_stream(tokens[:tokens.find("### Instruction:")].strip())

                    state_chatbot[-1] = (
                        instruction_display, 
                        processed_response
                    )
                    yield (state_chatbot, state_chatbot, f"{context} {bot_summarized_response}".strip(), state_recieved_docs)
                    break
                else:
                    agg_tokens = ""
                    cutoff_idx = 0

        if agg_tokens == "":
            processed_response, to_exit = post_process_stream(tokens)
            state_chatbot[-1] = (instruction_display, processed_response)
            yield (state_chatbot, state_chatbot, f"{context} {bot_summarized_response}".strip(), state_recieved_docs)

            if to_exit:
                break

        prev_index = len(tokens)
    if no_recent_retrieve:
        state_chatbot_copy = state_chatbot[:]
        state_chatbot_copy[-1][1] += "\n"+PLEASE_SPECIFY
        yield (
            state_chatbot,
            state_chatbot_copy,
            f"{context} {bot_summarized_response}".strip(),
            state_recieved_docs
        )
    else:
        yield (
            state_chatbot,
            state_chatbot,
            f"{context} {bot_summarized_response}".strip(),
            state_recieved_docs
        )


def chat_batch(
    contexts,
    instructions, 
    state_chatbots,
):
    state_results = []
    ctx_results = []

    instruct_prompts = [
        generate_prompt(instruct, histories, ctx)[0]
        for ctx, instruct, histories in zip(contexts, instructions, state_chatbots)
    ]
        
    bot_responses = get_output_batch(
        global_vars.model, global_vars.tokenizer, instruct_prompts, global_vars.gen_config
    )
    bot_responses = post_processes_batch(bot_responses)

    for ctx, instruction, bot_response, state_chatbot in zip(contexts, instructions, bot_responses, state_chatbots):
        new_state_chatbot = state_chatbot + [('' if instruction == SPECIAL_STRS["continue"] else instruction, bot_response)]
        ctx_results.append(gr.Textbox.update(value=bot_response) if instruction == SPECIAL_STRS["summarize"] else ctx)
        state_results.append(new_state_chatbot)

    return (state_results, state_results, ctx_results)