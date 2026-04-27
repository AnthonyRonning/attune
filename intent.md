# Intent

The following is the user's original project intent, captured verbatim.

---

I would like to make a new greenfield project but want to brainstorm with you on the intent and reasoning and potential solution / architecture of it. 

So I've worked with open source models for a very long time. small ones like gemma and big ones like kimi k2.5 or glm 5. so i've seen a wide range of behavior and edge cases. 

the BIGGEST issue I see happens out of my control as an application developer and completions api consumer - this is typically solely around tool calling / chat template / parsing at the inference engine level, often with vllm. 

its riddled with bugs and is often neglected by chinese labs due to it them running their own custom closed source engines and offering through the api so they don't have a lot of incentives to helping fix these bugs. 

what i find is that these models can often be very intelligent! which is such a shame that they are getting caught up and stuck on tool calls and random parsing. they are intelligent beings that are being limited by the deterministic chat templates and it really sucks. 

over the past year I've gotten really familiar with the dspy model. I love it! THROW AWAY THE inference engine level tool calls that everyone has revolved around!! xml-style, with a tight integration with prompt optimizers like gepa. a big focus on datasets and learning how to prompt each of these models at an individual level so that we can send them prompts they do well on. instead of forcing a square through a circle hole. no more getting cut off at the vllm level and getting something generic from the completions api like "sure! let me read that file to understand more" but no tool call, and stop_reason = done or something. did the model forget to generate the tool call? NO! it clearly was going to, it knew that! you know what happened? vllm saw a tool format it couldn't auto parse and then "safely failed" - but this is BAD! it leads to agentic ai stopping unneccessary. in this age of autonomous ai, silently stopping is bad. 

i have personally and professionally seen amazing results in my dspy agents. for one, they dont have problems with these xml tools in their normal content field. even calling consequtive tool calls in one shot. it all works great. they know how to do it well. any structure json issue that dspy can't resolve (they ahve a really good parser) i actually send some of the conversation history, and the "malformed" response to a correction agent to say something like "you are a correction agent, fix this models response in order to adhere to the structure and available tools..." (simplified). i can also fine tune model behavior with preferences too. and all of this i can put into datasets and gepa optimize it. and also do it whenever we change models! it's brilliant. 

there's one problem though. 

these are all custom agents and applications i make. i NEVER get to take advantage of my opinionated prompting style because the whole ecosystem of applications just do standard prompting and standard tool calling. so when i try to use open source models with these tools, there's two problems. one is the prompts are typically made against very powerful models, and very specific ones. so what prompt works well with one, it messes up on the other. that's just a quirk across all and any models. the second issue is the whole spill i had about tool calling. so i never really get to use open models in any of these open source agent ecosystems. coding agents, programs, etc. etc. 

so this leads me to the solution i want: 

a completions api proxy that intercepts prompts and the responses, and helps "fix" any issues live before sending them back to the caller. think of it like an openrouter that does prompt optimization and response correcting. i think there's different scenarios here we can do. like we don't really have control over the phrasing or intents that the applications or users send through completions through this proxy. they will have their own prompts and intents and styles and tools, etc. so it would have a hard time _changing_ the prompt side of it. but we at least see what the prompts and tools available are. a later phase of this project could have some sort of optional xml-style conversion from the standard json tools parameters (and conversly conversion from what the model responds with and back to standard tool response that the completions caller expects). that could be an optional phase 2. but the phase 1 is to get the models to respond correctly, and correct things as we go. 

there's a few technical ways i have opinions about how to accomplish this, but i'm opened to discussing the best methods. for instance i think we should just not stream back to the client so we can actually control its reponses to the client. that is a good first phase. if we see a few things we can try to judge/correct/regenerate: 
1. "let me do X" yet stops -> either do something like retry, or say "continue", or try to run some sort of correction agent to guess what it was trying to do to artificialy resume.
2. content tokens in thinking/reasoning -> fix it by outputting it as content instead
3. json tool call is slightly wrong -> deterministically try to detect and fix, if still fails -> correction agent tries to correct, if its completely hallucinated and doesn't exist at all, either retry or return the hallucinated tool and let application layer fail it
4. stuff like that. i'm sure we'll find many cases over time. the gist is that we "hold" the response and only after getting the complete response, try to correct if needed, then allow it to stream back to the completions api. 

we can do more advanced stuff like there, like have a reasoning summarizer to provide artificiatial thinking tokens stream back to the client while all of this is going on, similar to how chatgpt/claude work where the thinking tokens are just summarizes anyways. but that's advanced and a nice to have. i really want to focus on the core proxy architecture that is capable of doing this. and would like to use dsrs (dspy in rust) and gepa for all of the inner correction agents, judges, etc. which may use the same model or a bigger judge model, but that's the gist. 

wdyt?
