import OpenAI from 'openai';
import { ReactNode } from 'react';
import { z } from 'zod';
import { LanguageModelV1FinishReason, LanguageModelV1CallWarning, LanguageModelV1 } from '@ai-sdk/provider';

type AIAction<T = any, R = any> = (...args: T[]) => Promise<R>;
type AIActions<T = any, R = any> = Record<string, AIAction<T, R>>;
type AIProviderProps<AIState = any, UIState = any, Actions = any> = {
    children: React.ReactNode;
    initialAIState?: AIState;
    initialUIState?: UIState;
    /** $ActionTypes is only added for type inference and is never used at runtime **/
    $ActionTypes?: Actions;
};
type AIProvider<AIState = any, UIState = any, Actions = any> = (props: AIProviderProps<AIState, UIState, Actions>) => Promise<React.ReactElement>;
type InferAIState<T, Fallback> = T extends AIProvider<infer AIState, any, any> ? AIState : Fallback;
type OnSetAIState<S> = ({ key, state, done, }: {
    key: string | number | symbol | undefined;
    state: S;
    done: boolean;
}) => void | Promise<void>;
type OnGetUIState<S> = AIAction<void, S | undefined>;
type ValueOrUpdater<T> = T | ((current: T) => T);
type MutableAIState<AIState> = {
    get: () => AIState;
    update: (newState: ValueOrUpdater<AIState>) => void;
    done: ((newState: AIState) => void) | (() => void);
};
declare const __internal_curr: unique symbol;
declare const __internal_error: unique symbol;
/**
 * StreamableValue is a value that can be streamed over the network via AI Actions.
 * To read the streamed values, use the `readStreamableValue` or `useStreamableValue` APIs.
 */
type StreamableValue<T = any, E = any> = {
    [__internal_curr]?: T;
    [__internal_error]?: E;
};

/**
 * Get the current AI state.
 * If `key` is provided, it will return the value of the specified key in the
 * AI state, if it's an object. If it's not an object, it will throw an error.
 *
 * @example const state = getAIState() // Get the entire AI state
 * @example const field = getAIState('key') // Get the value of the key
 */
declare function getAIState<AI extends AIProvider = any>(): Readonly<InferAIState<AI, any>>;
declare function getAIState<AI extends AIProvider = any>(key: keyof InferAIState<AI, any>): Readonly<InferAIState<AI, any>[typeof key]>;
/**
 * Get the mutable AI state. Note that you must call `.done()` when finishing
 * updating the AI state.
 *
 * @example
 * ```tsx
 * const state = getMutableAIState()
 * state.update({ ...state.get(), key: 'value' })
 * state.update((currentState) => ({ ...currentState, key: 'value' }))
 * state.done()
 * ```
 *
 * @example
 * ```tsx
 * const state = getMutableAIState()
 * state.done({ ...state.get(), key: 'value' }) // Done with a new state
 * ```
 */
declare function getMutableAIState<AI extends AIProvider = any>(): MutableAIState<InferAIState<AI, any>>;
declare function getMutableAIState<AI extends AIProvider = any>(key: keyof InferAIState<AI, any>): MutableAIState<InferAIState<AI, any>[typeof key]>;

type StreamableUIWrapper = {
    /**
     * The value of the streamable UI. This can be returned from a Server Action and received by the client.
     */
    readonly value: React.ReactNode;
    /**
     * This method updates the current UI node. It takes a new UI node and replaces the old one.
     */
    update(value: React.ReactNode): StreamableUIWrapper;
    /**
     * This method is used to append a new UI node to the end of the old one.
     * Once appended a new UI node, the previous UI node cannot be updated anymore.
     *
     * @example
     * ```jsx
     * const ui = createStreamableUI(<div>hello</div>)
     * ui.append(<div>world</div>)
     *
     * // The UI node will be:
     * // <>
     * //   <div>hello</div>
     * //   <div>world</div>
     * // </>
     * ```
     */
    append(value: React.ReactNode): StreamableUIWrapper;
    /**
     * This method is used to signal that there is an error in the UI stream.
     * It will be thrown on the client side and caught by the nearest error boundary component.
     */
    error(error: any): StreamableUIWrapper;
    /**
     * This method marks the UI node as finalized. You can either call it without any parameters or with a new UI node as the final state.
     * Once called, the UI node cannot be updated or appended anymore.
     *
     * This method is always **required** to be called, otherwise the response will be stuck in a loading state.
     */
    done(...args: [React.ReactNode] | []): StreamableUIWrapper;
};
/**
 * Create a piece of changeable UI that can be streamed to the client.
 * On the client side, it can be rendered as a normal React node.
 */
declare function createStreamableUI(initialValue?: React.ReactNode): StreamableUIWrapper;
/**
 * Create a wrapped, changeable value that can be streamed to the client.
 * On the client side, the value can be accessed via the readStreamableValue() API.
 */
declare function createStreamableValue<T = any, E = any>(initialValue?: T | ReadableStream<T>): StreamableValueWrapper<T, E>;
type StreamableValueWrapper<T, E> = {
    /**
     * The value of the streamable. This can be returned from a Server Action and
     * received by the client. To read the streamed values, use the
     * `readStreamableValue` or `useStreamableValue` APIs.
     */
    readonly value: StreamableValue<T, E>;
    /**
     * This method updates the current value with a new one.
     */
    update(value: T): StreamableValueWrapper<T, E>;
    /**
     * This method is used to append a delta string to the current value. It
     * requires the current value of the streamable to be a string.
     *
     * @example
     * ```jsx
     * const streamable = createStreamableValue('hello');
     * streamable.append(' world');
     *
     * // The value will be 'hello world'
     * ```
     */
    append(value: T): StreamableValueWrapper<T, E>;
    /**
     * This method is used to signal that there is an error in the value stream.
     * It will be thrown on the client side when consumed via
     * `readStreamableValue` or `useStreamableValue`.
     */
    error(error: any): StreamableValueWrapper<T, E>;
    /**
     * This method marks the value as finalized. You can either call it without
     * any parameters or with a new value as the final state.
     * Once called, the value cannot be updated or appended anymore.
     *
     * This method is always **required** to be called, otherwise the response
     * will be stuck in a loading state.
     */
    done(...args: [T] | []): StreamableValueWrapper<T, E>;
};

type Streamable$2 = ReactNode | Promise<ReactNode>;
type Renderer$2<T> = (props: T) => Streamable$2 | Generator<Streamable$2, Streamable$2, void> | AsyncGenerator<Streamable$2, Streamable$2, void>;
/**
 * `render` is a helper function to create a streamable UI from some LLMs.
 * This API only supports OpenAI's GPT models with Function Calling and Assistants Tools,
 * please use `streamUI` for compatibility with other providers.
 *
 * @deprecated It's recommended to use the `streamUI` API for compatibility with AI SDK Core APIs
 * and future features. This API will be removed in a future release.
 */
declare function render<TS extends {
    [name: string]: z.Schema;
} = {}, FS extends {
    [name: string]: z.Schema;
} = {}>(options: {
    /**
     * The model name to use. Must be OpenAI SDK compatible. Tools and Functions are only supported
     * GPT models (3.5/4), OpenAI Assistants, Mistral small and large, and Fireworks firefunction-v1.
     *
     * @example "gpt-3.5-turbo"
     */
    model: string;
    /**
     * The provider instance to use. Currently the only provider available is OpenAI.
     * This needs to match the model name.
     */
    provider: OpenAI;
    messages: Parameters<typeof OpenAI.prototype.chat.completions.create>[0]['messages'];
    text?: Renderer$2<{
        /**
         * The full text content from the model so far.
         */
        content: string;
        /**
         * The new appended text content from the model since the last `text` call.
         */
        delta: string;
        /**
         * Whether the model is done generating text.
         * If `true`, the `content` will be the final output and this call will be the last.
         */
        done: boolean;
    }>;
    tools?: {
        [name in keyof TS]: {
            description?: string;
            parameters: TS[name];
            render: Renderer$2<z.infer<TS[name]>>;
        };
    };
    functions?: {
        [name in keyof FS]: {
            description?: string;
            parameters: FS[name];
            render: Renderer$2<z.infer<FS[name]>>;
        };
    };
    initial?: ReactNode;
    temperature?: number;
}): ReactNode;

type CallSettings = {
    /**
  Maximum number of tokens to generate.
     */
    maxTokens?: number;
    /**
  Temperature setting. This is a number between 0 (almost no randomness) and
  1 (very random).
  
  It is recommended to set either `temperature` or `topP`, but not both.
  
  @default 0
     */
    temperature?: number;
    /**
  Nucleus sampling. This is a number between 0 and 1.
  
  E.g. 0.1 would mean that only tokens with the top 10% probability mass
  are considered.
  
  It is recommended to set either `temperature` or `topP`, but not both.
     */
    topP?: number;
    /**
  Only sample from the top K options for each subsequent token.
  
  Used to remove "long tail" low probability responses.
  Recommended for advanced use cases only. You usually only need to use temperature.
     */
    topK?: number;
    /**
  Presence penalty setting. It affects the likelihood of the model to
  repeat information that is already in the prompt.
  
  The presence penalty is a number between -1 (increase repetition)
  and 1 (maximum penalty, decrease repetition). 0 means no penalty.
  
  @default 0
     */
    presencePenalty?: number;
    /**
  Frequency penalty setting. It affects the likelihood of the model
  to repeatedly use the same words or phrases.
  
  The frequency penalty is a number between -1 (increase repetition)
  and 1 (maximum penalty, decrease repetition). 0 means no penalty.
  
  @default 0
     */
    frequencyPenalty?: number;
    /**
  Stop sequences.
  If set, the model will stop generating text when one of the stop sequences is generated.
  Providers may have limits on the number of stop sequences.
     */
    stopSequences?: string[];
    /**
  The seed (integer) to use for random sampling. If set and supported
  by the model, calls will generate deterministic results.
     */
    seed?: number;
    /**
  Maximum number of retries. Set to 0 to disable retries.
  
  @default 2
     */
    maxRetries?: number;
    /**
  Abort signal.
     */
    abortSignal?: AbortSignal;
    /**
  Additional HTTP headers to be sent with the request.
  Only applicable for HTTP-based providers.
     */
    headers?: Record<string, string | undefined>;
};

/**
Data content. Can either be a base64-encoded string, a Uint8Array, an ArrayBuffer, or a Buffer.
 */
type DataContent = string | Uint8Array | ArrayBuffer | Buffer;

/**
Text content part of a prompt. It contains a string of text.
 */
interface TextPart {
    type: 'text';
    /**
  The text content.
     */
    text: string;
}
/**
Image content part of a prompt. It contains an image.
 */
interface ImagePart {
    type: 'image';
    /**
  Image data. Can either be:
  
  - data: a base64-encoded string, a Uint8Array, an ArrayBuffer, or a Buffer
  - URL: a URL that points to the image
     */
    image: DataContent | URL;
    /**
  Optional mime type of the image.
     */
    mimeType?: string;
}
/**
Tool call content part of a prompt. It contains a tool call (usually generated by the AI model).
 */
interface ToolCallPart {
    type: 'tool-call';
    /**
  ID of the tool call. This ID is used to match the tool call with the tool result.
   */
    toolCallId: string;
    /**
  Name of the tool that is being called.
   */
    toolName: string;
    /**
  Arguments of the tool call. This is a JSON-serializable object that matches the tool's input schema.
     */
    args: unknown;
}
/**
Tool result content part of a prompt. It contains the result of the tool call with the matching ID.
 */
interface ToolResultPart {
    type: 'tool-result';
    /**
  ID of the tool call that this result is associated with.
   */
    toolCallId: string;
    /**
  Name of the tool that generated this result.
    */
    toolName: string;
    /**
  Result of the tool call. This is a JSON-serializable object.
     */
    result: unknown;
    /**
  Optional flag if the result is an error or an error message.
     */
    isError?: boolean;
}

/**
A message that can be used in the `messages` field of a prompt.
It can be a user message, an assistant message, or a tool message.
 */
type CoreMessage = CoreSystemMessage | CoreUserMessage | CoreAssistantMessage | CoreToolMessage;
/**
 A system message. It can contain system information.

 Note: using the "system" part of the prompt is strongly preferred
 to increase the resilience against prompt injection attacks,
 and because not all providers support several system messages.
 */
type CoreSystemMessage = {
    role: 'system';
    content: string;
};
/**
A user message. It can contain text or a combination of text and images.
 */
type CoreUserMessage = {
    role: 'user';
    content: UserContent;
};
/**
Content of a user message. It can be a string or an array of text and image parts.
 */
type UserContent = string | Array<TextPart | ImagePart>;
/**
An assistant message. It can contain text, tool calls, or a combination of text and tool calls.
 */
type CoreAssistantMessage = {
    role: 'assistant';
    content: AssistantContent;
};
/**
Content of an assistant message. It can be a string or an array of text and tool call parts.
 */
type AssistantContent = string | Array<TextPart | ToolCallPart>;
/**
A tool message. It contains the result of one or more tool calls.
 */
type CoreToolMessage = {
    role: 'tool';
    content: ToolContent;
};
/**
Content of a tool message. It is an array of tool result parts.
 */
type ToolContent = Array<ToolResultPart>;

/**
Prompt part of the AI function options. It contains a system message, a simple text prompt, or a list of messages.
 */
type Prompt = {
    /**
  System message to include in the prompt. Can be used with `prompt` or `messages`.
     */
    system?: string;
    /**
  A simple text prompt. You can either use `prompt` or `messages` but not both.
   */
    prompt?: string;
    /**
  A list of messsages. You can either use `prompt` or `messages` but not both.
     */
    messages?: Array<CoreMessage>;
};

/**
Represents the number of tokens used in a prompt and completion.
 */
type CompletionTokenUsage = {
    /**
  The number of tokens used in the prompt.
     */
    promptTokens: number;
    /**
  The number of tokens used in the completion.
   */
    completionTokens: number;
    /**
  The total number of tokens used (promptTokens + completionTokens).
     */
    totalTokens: number;
};

/**
Reason why a language model finished generating a response.

Can be one of the following:
- `stop`: model generated stop sequence
- `length`: model generated maximum number of tokens
- `content-filter`: content filter violation stopped the model
- `tool-calls`: model triggered tool calls
- `error`: model stopped because of an error
- `other`: model stopped for other reasons
*/
type FinishReason$1 = LanguageModelV1FinishReason;
/**
Warning from the model provider for this call. The call will proceed, but e.g.
some settings might not be supported, which can lead to suboptimal results.
*/
type CallWarning = LanguageModelV1CallWarning;
/**
Tool choice for the generation. It supports the following settings:

- `auto` (default): the model can choose whether and which tools to call.
- `required`: the model must call a tool. It can choose which tool to call.
- `none`: the model must not call tools
- `{ type: 'tool', toolName: string (typed) }`: the model must call the specified tool
 */
type CoreToolChoice<TOOLS extends Record<string, unknown>> = 'auto' | 'none' | 'required' | {
    type: 'tool';
    toolName: keyof TOOLS;
};

type Streamable$1 = ReactNode | Promise<ReactNode>;
type Renderer$1<T extends Array<any>> = (...args: T) => Streamable$1 | Generator<Streamable$1, Streamable$1, void> | AsyncGenerator<Streamable$1, Streamable$1, void>;
type RenderTool<PARAMETERS extends z.ZodTypeAny = any> = {
    description?: string;
    parameters: PARAMETERS;
    generate?: Renderer$1<[
        z.infer<PARAMETERS>,
        {
            toolName: string;
            toolCallId: string;
        }
    ]>;
};
type RenderText$1 = Renderer$1<[
    {
        /**
         * The full text content from the model so far.
         */
        content: string;
        /**
         * The new appended text content from the model since the last `text` call.
         */
        delta: string;
        /**
         * Whether the model is done generating text.
         * If `true`, the `content` will be the final output and this call will be the last.
         */
        done: boolean;
    }
]>;
type RenderResult$1 = {
    value: ReactNode;
} & Awaited<ReturnType<LanguageModelV1['doStream']>>;
/**
 * `streamUI` is a helper function to create a streamable UI from LLMs.
 */
declare function streamUI<TOOLS extends {
    [name: string]: z.ZodTypeAny;
} = {}>({ model, tools, toolChoice, system, prompt, messages, maxRetries, abortSignal, headers, initial, text, onFinish, ...settings }: CallSettings & Prompt & {
    /**
     * The language model to use.
     */
    model: LanguageModelV1;
    /**
     * The tools that the model can call. The model needs to support calling tools.
     */
    tools?: {
        [name in keyof TOOLS]: RenderTool<TOOLS[name]>;
    };
    /**
     * The tool choice strategy. Default: 'auto'.
     */
    toolChoice?: CoreToolChoice<TOOLS>;
    text?: RenderText$1;
    initial?: ReactNode;
    /**
     * Callback that is called when the LLM response and the final object validation are finished.
     */
    onFinish?: (event: {
        /**
         * The reason why the generation finished.
         */
        finishReason: FinishReason$1;
        /**
         * The token usage of the generated response.
         */
        usage: CompletionTokenUsage;
        /**
         * The final ui node that was generated.
         */
        value: ReactNode;
        /**
         * Warnings from the model provider (e.g. unsupported settings)
         */
        warnings?: CallWarning[];
        /**
         * Optional raw response data.
         */
        rawResponse?: {
            /**
             * Response headers.
             */
            headers?: Record<string, string>;
        };
    }) => Promise<void> | void;
}): Promise<RenderResult$1>;

type FinishReason = 'stop' | 'length' | 'content-filter' | 'tool-calls' | 'error' | 'other' | 'unknown';
type Streamable = ReactNode | Promise<ReactNode>;
type Renderer<T extends Array<any>> = (...args: T) => Streamable | Generator<Streamable, Streamable, void> | AsyncGenerator<Streamable, Streamable, void>;
type RenderText = Renderer<[
    {
        /**
         * The full text content from the model so far.
         */
        content: string;
        /**
         * The new appended text content from the model since the last `text` call.
         */
        delta: string;
        /**
         * Whether the model is done generating text.
         * If `true`, the `content` will be the final output and this call will be the last.
         */
        done: boolean;
    }
]>;
type RenderResult = {
    value: ReactNode;
} & Awaited<ReturnType<LanguageModelV1['doStream']>>;
/**
 * `streamUI` is a helper function to create a streamable UI from LLMs.
 */
declare function streamUIWithProcess<TOOLS extends {
    [name: string]: z.ZodTypeAny;
} = {}>({ processUrl, body, maxRetries, abortSignal, headers, initial, text, onFinish, ...settings }: CallSettings & {
    processUrl: string;
    body: z.AnyZodObject;
    text?: RenderText;
    initial?: ReactNode;
    /**
     * Callback that is called when the LLM response and the final object validation are finished.
     */
    onFinish?: (event: {
        /**
         * The reason why the generation finished.
         */
        finishReason: FinishReason;
        /**
         * The token usage of the generated response.
         */
        usage: CompletionTokenUsage;
        /**
         * The final ui node that was generated.
         */
        value: ReactNode;
        /**
         * Warnings from the model provider (e.g. unsupported settings)
         */
        warnings?: CallWarning[];
        /**
         * Optional raw response data.
         */
        rawResponse?: {
            /**
             * Response headers.
             */
            headers?: Record<string, string>;
        };
    }) => Promise<void> | void;
}): Promise<RenderResult>;

declare function createAI<AIState = any, UIState = any, Actions extends AIActions = {}>({ actions, initialAIState, initialUIState, onSetAIState, onGetUIState, }: {
    actions: Actions;
    initialAIState?: AIState;
    initialUIState?: UIState;
    /**
     * This function is called whenever the AI state is updated by an Action.
     * You can use this to persist the AI state to a database, or to send it to a
     * logging service.
     */
    onSetAIState?: OnSetAIState<AIState>;
    /**
     * This function is used to retrieve the UI state based on the AI state.
     * For example, to render the initial UI state based on a given AI state, or
     * to sync the UI state when the application is already loaded.
     *
     * If returning `undefined`, the client side UI state will not be updated.
     *
     * This function must be annotated with the `"use server"` directive.
     *
     * @example
     * ```tsx
     * onGetUIState: async () => {
     *   'use server';
     *
     *   const currentAIState = getAIState();
     *   const externalAIState = await loadAIStateFromDatabase();
     *
     *   if (currentAIState === externalAIState) return undefined;
     *
     *   // Update current AI state and return the new UI state
     *   const state = getMutableAIState()
     *   state.done(externalAIState)
     *
     *   return <div>...</div>;
     * }
     * ```
     */
    onGetUIState?: OnGetUIState<UIState>;
}): AIProvider<AIState, UIState, Actions>;

export { createAI, createStreamableUI, createStreamableValue, getAIState, getMutableAIState, render, streamUI, streamUIWithProcess };
