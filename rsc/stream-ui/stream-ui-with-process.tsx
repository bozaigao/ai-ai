import { LanguageModelV1, LanguageModelV1StreamPart } from '@ai-sdk/provider';
import {
  ParseResult,
  createEventSourceResponseHandler,
  createJsonErrorResponseHandler,
  postJsonToApi,
} from '@ai-sdk/provider-utils';
import { ReactNode } from 'react';
import { z } from 'zod';
import { CallSettings } from '../../core/prompt/call-settings';
import { Prompt } from '../../core/prompt/prompt';
import { CallWarning } from '../../core/types';
import {
  CompletionTokenUsage,
  calculateCompletionTokenUsage,
} from '../../core/types/token-usage';
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
  statusCode: z.number(),
  body: z.string(),
});

export const ErrorDataSchema = z.object({
  error: z.object({
    success: z.boolean(),
    statusCode: z.number(),
    msg: z.string(),
    data: z.any().nullable(),
  }),
});

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
  processUrl,
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
    processUrl: string;
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

  const question =
    messages && messages.length > 0
      ? messages[messages.length - 1].content
      : 'hello';
  const retry = retryWithExponentialBackoff({ maxRetries });
  const result = await retry(async () => {
    const { value: response } = await postJsonToApi({
      url: processUrl,
      body: {
        prompt: question,
        stream: true,
      },
      failedResponseHandler: createJsonErrorResponseHandler({
        errorSchema: ErrorDataSchema,
        errorToMessage: data => data.error.msg,
      }),
      successfulResponseHandler:
      createEventSourceResponseHandlerForProgress(ResponseSchema),
    });

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
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof ResponseSchema>>,
          LanguageModelV1StreamPart
        >({
          transform(chunk: any, controller) {
            const res = JSON.parse(chunk);
            // handle failed chunk parsing / validation:
            if (res.statusCode !== 200) {
              finishReason = 'error';
              controller.enqueue({ type: 'error', error: chunk.msg });
              return;
            }
            controller.enqueue({
              type: 'text-delta',
              textDelta: JSON.stringify(res.body.replace(/\n/g, '')),
            });
          },
        }),
      ),
      rawCall: { rawPrompt: [], rawSettings: { tools: [] } },
      rawResponse: { headers: {} },
      warnings: [],
    };

    return result;
  });

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
