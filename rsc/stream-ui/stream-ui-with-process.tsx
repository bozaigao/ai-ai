import { InvalidResponseDataError, LanguageModelV1 } from '@ai-sdk/provider';
import {
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  generateId,
  isParsableJson,
  postJsonToApi,
  safeParseJSON,
} from '@ai-sdk/provider-utils';
import { ReactNode } from 'react';
import { z } from 'zod';
import { CallSettings } from '../../core/prompt/call-settings';
import { convertToLanguageModelPrompt } from '../../core/prompt/convert-to-language-model-prompt';
import { getValidatedPrompt } from '../../core/prompt/get-validated-prompt';
import { prepareCallSettings } from '../../core/prompt/prepare-call-settings';
import { prepareToolsAndToolChoice } from '../../core/prompt/prepare-tools-and-tool-choice';
import { Prompt } from '../../core/prompt/prompt';
import { CallWarning, CoreToolChoice } from '../../core/types';
import {
  CompletionTokenUsage,
  calculateCompletionTokenUsage,
} from '../../core/types/token-usage';
import { InvalidToolArgumentsError } from '../../errors/invalid-tool-arguments-error';
import { NoSuchToolError } from '../../errors/no-such-tool-error';
import { createResolvablePromise } from '../../util/create-resolvable-promise';
import { isAsyncGenerator } from '../../util/is-async-generator';
import { isGenerator } from '../../util/is-generator';
import { retryWithExponentialBackoff } from '../../util/retry-with-exponential-backoff';
import { createStreamableUI } from '../streamable';

type FinishReason =
  | 'stop'
  | 'length'
  | 'content-filter'
  | 'tool-calls'
  | 'error'
  | 'other'
  | 'unknown';

const ResponseSchema = z.object({
  success: z.boolean(),
  code: z.string(),
  msg: z.string(),
  data: z.array(
    z.any().nullable(),
  ),
});

export const ErrorDataSchema = z.object({
  error: z.object({
    success: z.boolean(),
    code: z.string(),
    msg: z.string(),
    data: z.any().nullable(),
  }),
});

export function mapOpenAIFinishReason(
  finishReason: string | null | undefined,
): FinishReason {
  switch (finishReason) {
    case 'stop':
      return 'stop';
    case 'length':
      return 'length';
    case 'content_filter':
      return 'content-filter';
    case 'function_call':
    case 'tool_calls':
      return 'tool-calls';
    default:
      return 'unknown';
  }
}

type Streamable = ReactNode | Promise<ReactNode>;

type Renderer<T extends Array<any>> = (
  ...args: T
) =>
  | Streamable
  | Generator<Streamable, Streamable, void>
  | AsyncGenerator<Streamable, Streamable, void>;

type RenderTool<PARAMETERS extends z.ZodTypeAny = any> = {
  description?: string;
  parameters: PARAMETERS;
  generate?: Renderer<
    [
      z.infer<PARAMETERS>,
      {
        toolName: string;
        toolCallId: string;
      },
    ]
  >;
};

type RenderText = Renderer<
  [
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
    },
  ]
>;

type RenderResult = {
  value: ReactNode;
} & Awaited<ReturnType<LanguageModelV1['doStream']>>;

const defaultTextRenderer: RenderText = ({ content }: { content: string }) =>
  content;

/**
 * `streamUI` is a helper function to create a streamable UI from LLMs.
 */
export async function streamUIWithProcess<
  TOOLS extends { [name: string]: z.ZodTypeAny } = {},
>({
  tools,
  toolChoice,
  prompt,
  messages,
  maxRetries,
  abortSignal,
  headers,
  initial,
  text,
  onFinish,
  ...settings
}: CallSettings &
  Prompt & {
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
  }): Promise<RenderResult> {
  if ('functions' in settings) {
    throw new Error(
      '`functions` is not supported in `streamUI`, use `tools` instead.',
    );
  }
  if ('provider' in settings) {
    throw new Error(
      '`provider` is no longer needed in `streamUI`. Use `model` instead.',
    );
  }
  if (tools) {
    for (const [name, tool] of Object.entries(tools)) {
      if ('render' in tool) {
        throw new Error(
          'Tool definition in `streamUI` should not have `render` property. Use `generate` instead. Found in tool: ' +
            name,
        );
      }
    }
  }

  const ui = createStreamableUI(initial);

  // The default text renderer just returns the content as string.
  const textRender = text || defaultTextRenderer;

  let finished: Promise<void> | undefined;

  async function render({
    args,
    renderer,
    streamableUI,
    isLastCall = false,
  }: {
    renderer: undefined | Renderer<any>;
    args: [payload: any] | [payload: any, options: any];
    streamableUI: ReturnType<typeof createStreamableUI>;
    isLastCall?: boolean;
  }) {
    if (!renderer) return;

    // create a promise that will be resolved when the render call is finished.
    // it is appended to the `finished` promise chain to ensure the render call
    // is finished before the next render call starts.
    const renderFinished = createResolvablePromise<void>();
    finished = finished
      ? finished.then(() => renderFinished.promise)
      : renderFinished.promise;

    const rendererResult = renderer(...args);

    if (isAsyncGenerator(rendererResult) || isGenerator(rendererResult)) {
      while (true) {
        const { done, value } = await rendererResult.next();
        const node = await value;

        if (isLastCall && done) {
          streamableUI.done(node);
        } else {
          streamableUI.update(node);
        }

        if (done) break;
      }
    } else {
      const node = await rendererResult;

      if (isLastCall) {
        streamableUI.done(node);
      } else {
        streamableUI.update(node);
      }
    }

    // resolve the promise to signal that the render call is finished
    renderFinished.resolve(undefined);
  }

  const retry = retryWithExponentialBackoff({ maxRetries });
  // const result = await retry(async () =>
  //   model.doStream({
  //     mode: {
  //       type: 'regular',
  //       ...prepareToolsAndToolChoice({ tools, toolChoice }),
  //     },
  //     ...prepareCallSettings(settings),
  //     inputFormat: validatedPrompt.type,
  //     prompt: await convertToLanguageModelPrompt({
  //       prompt: validatedPrompt,
  //       modelSupportsImageUrls: model.supportsImageUrls,
  //     }),
  //     abortSignal,
  //     headers,
  //   }),
  // );

  const { value: response } = await postJsonToApi({
    url: 'https://0yjhl0kfcd.execute-api.us-east-1.amazonaws.com/spangle/prompt',
    body: { idType: 'product', idValue: ['191877631128'] },
    failedResponseHandler: createJsonErrorResponseHandler({
      errorSchema: ErrorDataSchema,
      errorToMessage: data => data.error.msg,
    }),
    successfulResponseHandler: createJsonResponseHandler(
      ResponseSchema,
    ),
  });

  console.log('😁prompt', JSON.stringify(response));
  let finishReason: FinishReason = 'other';
  let usage: { promptTokens: number; completionTokens: number } = {
    promptTokens: Number.NaN,
    completionTokens: Number.NaN,
  };
  const toolCalls: Array<{
    id: string;
    type: 'function';
    function: {
      name: string;
      arguments: string;
    };
  }> = [];

  const useLegacyFunctionCalling = true;

  const result = {
    //@ts-ignore
    stream: response.pipeThrough({
      transform(chunk: any, controller: any) {
        // handle failed chunk parsing / validation:
        if (!chunk.success) {
          finishReason = 'error';
          controller.enqueue({ type: 'error', error: chunk.error });
          return;
        }

        const value = chunk.value;

        // handle error chunks:
        if ('error' in value) {
          finishReason = 'error';
          controller.enqueue({ type: 'error', error: value.error });
          return;
        }

        if (value.usage != null) {
          usage = {
            promptTokens: value.usage.prompt_tokens,
            completionTokens: value.usage.completion_tokens,
          };
        }

        const choice = value.choices[0];

        if (choice?.finish_reason != null) {
          finishReason = mapOpenAIFinishReason(choice.finish_reason);
        }

        if (choice?.delta == null) {
          return;
        }

        const delta = choice.delta;

        if (delta.content != null) {
          controller.enqueue({
            type: 'text-delta',
            textDelta: delta.content,
          });
        }

        const mappedToolCalls: typeof delta.tool_calls =
          useLegacyFunctionCalling && delta.function_call != null
            ? [
                {
                  type: 'function',
                  id: generateId(),
                  function: delta.function_call,
                  index: 0,
                },
              ]
            : delta.tool_calls;

        if (mappedToolCalls != null) {
          for (const toolCallDelta of mappedToolCalls) {
            const index = toolCallDelta.index;

            // Tool call start. OpenAI returns all information except the arguments in the first chunk.
            if (toolCalls[index] == null) {
              if (toolCallDelta.type !== 'function') {
                throw new InvalidResponseDataError({
                  data: toolCallDelta,
                  message: `Expected 'function' type.`,
                });
              }

              if (toolCallDelta.id == null) {
                throw new InvalidResponseDataError({
                  data: toolCallDelta,
                  message: `Expected 'id' to be a string.`,
                });
              }

              if (toolCallDelta.function?.name == null) {
                throw new InvalidResponseDataError({
                  data: toolCallDelta,
                  message: `Expected 'function.name' to be a string.`,
                });
              }

              toolCalls[index] = {
                id: toolCallDelta.id,
                type: 'function',
                function: {
                  name: toolCallDelta.function.name,
                  arguments: toolCallDelta.function.arguments ?? '',
                },
              };

              const toolCall = toolCalls[index];

              // check if tool call is complete (some providers send the full tool call in one chunk)
              if (
                toolCall.function?.name != null &&
                toolCall.function?.arguments != null &&
                isParsableJson(toolCall.function.arguments)
              ) {
                // send delta
                controller.enqueue({
                  type: 'tool-call-delta',
                  toolCallType: 'function',
                  toolCallId: toolCall.id,
                  toolName: toolCall.function.name,
                  argsTextDelta: toolCall.function.arguments,
                });

                // send tool call
                controller.enqueue({
                  type: 'tool-call',
                  toolCallType: 'function',
                  toolCallId: toolCall.id ?? generateId(),
                  toolName: toolCall.function.name,
                  args: toolCall.function.arguments,
                });
              }

              continue;
            }

            // existing tool call, merge
            const toolCall = toolCalls[index];

            if (toolCallDelta.function?.arguments != null) {
              toolCall.function!.arguments +=
                toolCallDelta.function?.arguments ?? '';
            }

            // send delta
            controller.enqueue({
              type: 'tool-call-delta',
              toolCallType: 'function',
              toolCallId: toolCall.id,
              toolName: toolCall.function.name,
              argsTextDelta: toolCallDelta.function.arguments ?? '',
            });

            // check if tool call is complete
            if (
              toolCall.function?.name != null &&
              toolCall.function?.arguments != null &&
              isParsableJson(toolCall.function.arguments)
            ) {
              controller.enqueue({
                type: 'tool-call',
                toolCallType: 'function',
                toolCallId: toolCall.id ?? generateId(),
                toolName: toolCall.function.name,
                args: toolCall.function.arguments,
              });
            }
          }
        }
      },
    }),
    rawCall: { rawPrompt: [], rawSettings: { tools: [] } },
    rawResponse: { headers: {} },
    warnings: [],
  };

  // For the stream and consume it asynchronously:
  const [stream, forkedStream] = result.stream.tee();
  (async () => {
    try {
      let content = '';
      let hasToolCall = false;

      const reader = forkedStream.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        switch (value.type) {
          case 'text-delta': {
            content += value.textDelta;
            render({
              renderer: textRender,
              args: [{ content, done: false, delta: value.textDelta }],
              streamableUI: ui,
            });
            break;
          }

          case 'tool-call-delta': {
            hasToolCall = true;
            break;
          }

          case 'tool-call': {
            const toolName = value.toolName as keyof TOOLS & string;

            if (!tools) {
              throw new NoSuchToolError({ toolName });
            }

            const tool = tools[toolName];
            if (!tool) {
              throw new NoSuchToolError({
                toolName,
                availableTools: Object.keys(tools),
              });
            }

            hasToolCall = true;
            const parseResult = safeParseJSON({
              text: value.args,
              schema: tool.parameters,
            });

            if (parseResult.success === false) {
              throw new InvalidToolArgumentsError({
                toolName,
                toolArgs: value.args,
                cause: parseResult.error,
              });
            }

            render({
              renderer: tool.generate,
              args: [
                parseResult.value,
                {
                  toolName,
                  toolCallId: value.toolCallId,
                },
              ],
              streamableUI: ui,
              isLastCall: true,
            });

            break;
          }

          case 'error': {
            throw value.error;
          }

          case 'finish': {
            onFinish?.({
              finishReason: value.finishReason,
              usage: calculateCompletionTokenUsage(value.usage),
              value: ui.value,
              warnings: result.warnings,
              rawResponse: result.rawResponse,
            });
          }
        }
      }

      if (!hasToolCall) {
        render({
          renderer: textRender,
          args: [{ content, done: true }],
          streamableUI: ui,
          isLastCall: true,
        });
      }

      await finished;
    } catch (error) {
      // During the stream rendering, we don't want to throw the error to the
      // parent scope but only let the React's error boundary to catch it.
      ui.error(error);
    }
  })();

  return {
    ...result,
    stream,
    value: ui.value,
  };
}
