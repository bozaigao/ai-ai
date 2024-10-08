"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name9 in all)
    __defProp(target, name9, { get: all[name9], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// streams/index.ts
var streams_exports = {};
__export(streams_exports, {
  AISDKError: () => import_provider11.AISDKError,
  AIStream: () => AIStream,
  APICallError: () => import_provider11.APICallError,
  AWSBedrockAnthropicMessagesStream: () => AWSBedrockAnthropicMessagesStream,
  AWSBedrockAnthropicStream: () => AWSBedrockAnthropicStream,
  AWSBedrockCohereStream: () => AWSBedrockCohereStream,
  AWSBedrockLlama2Stream: () => AWSBedrockLlama2Stream,
  AWSBedrockStream: () => AWSBedrockStream,
  AnthropicStream: () => AnthropicStream,
  AssistantResponse: () => AssistantResponse,
  CohereStream: () => CohereStream,
  DownloadError: () => DownloadError,
  EmptyResponseBodyError: () => import_provider11.EmptyResponseBodyError,
  GoogleGenerativeAIStream: () => GoogleGenerativeAIStream,
  HuggingFaceStream: () => HuggingFaceStream,
  InkeepStream: () => InkeepStream,
  InvalidArgumentError: () => InvalidArgumentError,
  InvalidDataContentError: () => InvalidDataContentError,
  InvalidMessageRoleError: () => InvalidMessageRoleError,
  InvalidModelIdError: () => InvalidModelIdError,
  InvalidPromptError: () => import_provider11.InvalidPromptError,
  InvalidResponseDataError: () => import_provider11.InvalidResponseDataError,
  InvalidToolArgumentsError: () => InvalidToolArgumentsError,
  JSONParseError: () => import_provider11.JSONParseError,
  LangChainAdapter: () => langchain_adapter_exports,
  LangChainStream: () => LangChainStream,
  LoadAPIKeyError: () => import_provider11.LoadAPIKeyError,
  MistralStream: () => MistralStream,
  NoObjectGeneratedError: () => NoObjectGeneratedError,
  NoSuchModelError: () => NoSuchModelError,
  NoSuchProviderError: () => NoSuchProviderError,
  NoSuchToolError: () => NoSuchToolError,
  OpenAIStream: () => OpenAIStream,
  ReplicateStream: () => ReplicateStream,
  RetryError: () => RetryError,
  StreamData: () => StreamData2,
  StreamingTextResponse: () => StreamingTextResponse,
  TypeValidationError: () => import_provider11.TypeValidationError,
  UnsupportedFunctionalityError: () => import_provider11.UnsupportedFunctionalityError,
  convertDataContentToBase64String: () => convertDataContentToBase64String,
  convertDataContentToUint8Array: () => convertDataContentToUint8Array,
  convertToCoreMessages: () => convertToCoreMessages,
  convertUint8ArrayToText: () => convertUint8ArrayToText,
  cosineSimilarity: () => cosineSimilarity,
  createCallbacksTransformer: () => createCallbacksTransformer,
  createEventStreamTransformer: () => createEventStreamTransformer,
  createStreamDataTransformer: () => createStreamDataTransformer,
  embed: () => embed,
  embedMany: () => embedMany,
  experimental_AssistantResponse: () => experimental_AssistantResponse,
  experimental_StreamData: () => experimental_StreamData,
  experimental_createModelRegistry: () => experimental_createModelRegistry,
  experimental_createProviderRegistry: () => experimental_createProviderRegistry,
  experimental_generateObject: () => experimental_generateObject,
  experimental_generateText: () => experimental_generateText,
  experimental_streamObject: () => experimental_streamObject,
  experimental_streamText: () => experimental_streamText,
  formatStreamPart: () => import_ui_utils6.formatStreamPart,
  generateId: () => generateId2,
  generateObject: () => generateObject,
  generateText: () => generateText,
  jsonSchema: () => jsonSchema,
  nanoid: () => nanoid,
  parseComplexResponse: () => import_ui_utils6.parseComplexResponse,
  parseStreamPart: () => import_ui_utils6.parseStreamPart,
  readDataStream: () => import_ui_utils6.readDataStream,
  readableFromAsyncIterable: () => readableFromAsyncIterable,
  streamObject: () => streamObject,
  streamText: () => streamText,
  streamToResponse: () => streamToResponse,
  tool: () => tool,
  trimStartOfStreamHelper: () => trimStartOfStreamHelper
});
module.exports = __toCommonJS(streams_exports);
var import_ui_utils6 = require("@ai-sdk/ui-utils");
var import_provider_utils8 = require("@ai-sdk/provider-utils");

// util/retry-with-exponential-backoff.ts
var import_provider2 = require("@ai-sdk/provider");
var import_provider_utils = require("@ai-sdk/provider-utils");

// util/delay.ts
async function delay(delayInMs) {
  return new Promise((resolve) => setTimeout(resolve, delayInMs));
}

// util/retry-error.ts
var import_provider = require("@ai-sdk/provider");
var name = "AI_RetryError";
var marker = `vercel.ai.error.${name}`;
var symbol = Symbol.for(marker);
var _a;
var RetryError = class extends import_provider.AISDKError {
  constructor({
    message,
    reason,
    errors
  }) {
    super({ name, message });
    this[_a] = true;
    this.reason = reason;
    this.errors = errors;
    this.lastError = errors[errors.length - 1];
  }
  static isInstance(error) {
    return import_provider.AISDKError.hasMarker(error, marker);
  }
  /**
   * @deprecated use `isInstance` instead
   */
  static isRetryError(error) {
    return error instanceof Error && error.name === name && typeof error.reason === "string" && Array.isArray(error.errors);
  }
  /**
   * @deprecated Do not use this method. It will be removed in the next major version.
   */
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      reason: this.reason,
      lastError: this.lastError,
      errors: this.errors
    };
  }
};
_a = symbol;

// util/retry-with-exponential-backoff.ts
var retryWithExponentialBackoff = ({
  maxRetries = 2,
  initialDelayInMs = 2e3,
  backoffFactor = 2
} = {}) => async (f) => _retryWithExponentialBackoff(f, {
  maxRetries,
  delayInMs: initialDelayInMs,
  backoffFactor
});
async function _retryWithExponentialBackoff(f, {
  maxRetries,
  delayInMs,
  backoffFactor
}, errors = []) {
  try {
    return await f();
  } catch (error) {
    if ((0, import_provider_utils.isAbortError)(error)) {
      throw error;
    }
    if (maxRetries === 0) {
      throw error;
    }
    const errorMessage = (0, import_provider_utils.getErrorMessage)(error);
    const newErrors = [...errors, error];
    const tryNumber = newErrors.length;
    if (tryNumber > maxRetries) {
      throw new RetryError({
        message: `Failed after ${tryNumber} attempts. Last error: ${errorMessage}`,
        reason: "maxRetriesExceeded",
        errors: newErrors
      });
    }
    if (error instanceof Error && import_provider2.APICallError.isAPICallError(error) && error.isRetryable === true && tryNumber <= maxRetries) {
      await delay(delayInMs);
      return _retryWithExponentialBackoff(
        f,
        { maxRetries, delayInMs: backoffFactor * delayInMs, backoffFactor },
        newErrors
      );
    }
    if (tryNumber === 1) {
      throw error;
    }
    throw new RetryError({
      message: `Failed after ${tryNumber} attempts with non-retryable error: '${errorMessage}'`,
      reason: "errorNotRetryable",
      errors: newErrors
    });
  }
}

// core/telemetry/assemble-operation-name.ts
function assembleOperationName({
  operationName,
  telemetry
}) {
  return {
    "operation.name": `${operationName}${(telemetry == null ? void 0 : telemetry.functionId) != null ? ` ${telemetry.functionId}` : ""}`
  };
}

// core/telemetry/get-base-telemetry-attributes.ts
function getBaseTelemetryAttributes({
  model,
  settings,
  telemetry,
  headers
}) {
  var _a9;
  return {
    "ai.model.provider": model.provider,
    "ai.model.id": model.modelId,
    // settings:
    ...Object.entries(settings).reduce((attributes, [key, value]) => {
      attributes[`ai.settings.${key}`] = value;
      return attributes;
    }, {}),
    // special telemetry information
    "resource.name": telemetry == null ? void 0 : telemetry.functionId,
    "ai.telemetry.functionId": telemetry == null ? void 0 : telemetry.functionId,
    // add metadata as attributes:
    ...Object.entries((_a9 = telemetry == null ? void 0 : telemetry.metadata) != null ? _a9 : {}).reduce(
      (attributes, [key, value]) => {
        attributes[`ai.telemetry.metadata.${key}`] = value;
        return attributes;
      },
      {}
    ),
    // request headers
    ...Object.entries(headers != null ? headers : {}).reduce((attributes, [key, value]) => {
      if (value !== void 0) {
        attributes[`ai.request.headers.${key}`] = value;
      }
      return attributes;
    }, {})
  };
}

// core/telemetry/get-tracer.ts
var import_api = require("@opentelemetry/api");

// core/telemetry/noop-tracer.ts
var noopTracer = {
  startSpan() {
    return noopSpan;
  },
  startActiveSpan(name9, arg1, arg2, arg3) {
    if (typeof arg1 === "function") {
      return arg1(noopSpan);
    }
    if (typeof arg2 === "function") {
      return arg2(noopSpan);
    }
    if (typeof arg3 === "function") {
      return arg3(noopSpan);
    }
  }
};
var noopSpan = {
  spanContext() {
    return noopSpanContext;
  },
  setAttribute() {
    return this;
  },
  setAttributes() {
    return this;
  },
  addEvent() {
    return this;
  },
  addLink() {
    return this;
  },
  addLinks() {
    return this;
  },
  setStatus() {
    return this;
  },
  updateName() {
    return this;
  },
  end() {
    return this;
  },
  isRecording() {
    return false;
  },
  recordException() {
    return this;
  }
};
var noopSpanContext = {
  traceId: "",
  spanId: "",
  traceFlags: 0
};

// core/telemetry/get-tracer.ts
var testTracer = void 0;
function getTracer({ isEnabled }) {
  if (!isEnabled) {
    return noopTracer;
  }
  if (testTracer) {
    return testTracer;
  }
  return import_api.trace.getTracer("ai");
}

// core/telemetry/record-span.ts
var import_api2 = require("@opentelemetry/api");
function recordSpan({
  name: name9,
  tracer,
  attributes,
  fn,
  endWhenDone = true
}) {
  return tracer.startActiveSpan(name9, { attributes }, async (span) => {
    try {
      const result = await fn(span);
      if (endWhenDone) {
        span.end();
      }
      return result;
    } catch (error) {
      try {
        if (error instanceof Error) {
          span.recordException({
            name: error.name,
            message: error.message,
            stack: error.stack
          });
          span.setStatus({
            code: import_api2.SpanStatusCode.ERROR,
            message: error.message
          });
        } else {
          span.setStatus({ code: import_api2.SpanStatusCode.ERROR });
        }
      } finally {
        span.end();
      }
      throw error;
    }
  });
}

// core/telemetry/select-telemetry-attributes.ts
function selectTelemetryAttributes({
  telemetry,
  attributes
}) {
  return Object.entries(attributes).reduce((attributes2, [key, value]) => {
    if (value === void 0) {
      return attributes2;
    }
    if (typeof value === "object" && "input" in value && typeof value.input === "function") {
      if ((telemetry == null ? void 0 : telemetry.recordInputs) === false) {
        return attributes2;
      }
      const result = value.input();
      return result === void 0 ? attributes2 : { ...attributes2, [key]: result };
    }
    if (typeof value === "object" && "output" in value && typeof value.output === "function") {
      if ((telemetry == null ? void 0 : telemetry.recordOutputs) === false) {
        return attributes2;
      }
      const result = value.output();
      return result === void 0 ? attributes2 : { ...attributes2, [key]: result };
    }
    return { ...attributes2, [key]: value };
  }, {});
}

// core/embed/embed.ts
async function embed({
  model,
  value,
  maxRetries,
  abortSignal,
  headers,
  experimental_telemetry: telemetry
}) {
  var _a9;
  const baseTelemetryAttributes = getBaseTelemetryAttributes({
    model,
    telemetry,
    headers,
    settings: { maxRetries }
  });
  const tracer = getTracer({ isEnabled: (_a9 = telemetry == null ? void 0 : telemetry.isEnabled) != null ? _a9 : false });
  return recordSpan({
    name: "ai.embed",
    attributes: selectTelemetryAttributes({
      telemetry,
      attributes: {
        ...assembleOperationName({ operationName: "ai.embed", telemetry }),
        ...baseTelemetryAttributes,
        "ai.value": { input: () => JSON.stringify(value) }
      }
    }),
    tracer,
    fn: async (span) => {
      const retry = retryWithExponentialBackoff({ maxRetries });
      const { embedding, usage, rawResponse } = await retry(
        () => (
          // nested spans to align with the embedMany telemetry data:
          recordSpan({
            name: "ai.embed.doEmbed",
            attributes: selectTelemetryAttributes({
              telemetry,
              attributes: {
                ...assembleOperationName({
                  operationName: "ai.embed.doEmbed",
                  telemetry
                }),
                ...baseTelemetryAttributes,
                // specific settings that only make sense on the outer level:
                "ai.values": { input: () => [JSON.stringify(value)] }
              }
            }),
            tracer,
            fn: async (doEmbedSpan) => {
              var _a10;
              const modelResponse = await model.doEmbed({
                values: [value],
                abortSignal,
                headers
              });
              const embedding2 = modelResponse.embeddings[0];
              const usage2 = (_a10 = modelResponse.usage) != null ? _a10 : { tokens: NaN };
              doEmbedSpan.setAttributes(
                selectTelemetryAttributes({
                  telemetry,
                  attributes: {
                    "ai.embeddings": {
                      output: () => modelResponse.embeddings.map(
                        (embedding3) => JSON.stringify(embedding3)
                      )
                    },
                    "ai.usage.tokens": usage2.tokens
                  }
                })
              );
              return {
                embedding: embedding2,
                usage: usage2,
                rawResponse: modelResponse.rawResponse
              };
            }
          })
        )
      );
      span.setAttributes(
        selectTelemetryAttributes({
          telemetry,
          attributes: {
            "ai.embedding": { output: () => JSON.stringify(embedding) },
            "ai.usage.tokens": usage.tokens
          }
        })
      );
      return new DefaultEmbedResult({ value, embedding, usage, rawResponse });
    }
  });
}
var DefaultEmbedResult = class {
  constructor(options) {
    this.value = options.value;
    this.embedding = options.embedding;
    this.usage = options.usage;
    this.rawResponse = options.rawResponse;
  }
};

// core/util/split-array.ts
function splitArray(array, chunkSize) {
  if (chunkSize <= 0) {
    throw new Error("chunkSize must be greater than 0");
  }
  const result = [];
  for (let i = 0; i < array.length; i += chunkSize) {
    result.push(array.slice(i, i + chunkSize));
  }
  return result;
}

// core/embed/embed-many.ts
async function embedMany({
  model,
  values,
  maxRetries,
  abortSignal,
  headers,
  experimental_telemetry: telemetry
}) {
  var _a9;
  const baseTelemetryAttributes = getBaseTelemetryAttributes({
    model,
    telemetry,
    headers,
    settings: { maxRetries }
  });
  const tracer = getTracer({ isEnabled: (_a9 = telemetry == null ? void 0 : telemetry.isEnabled) != null ? _a9 : false });
  return recordSpan({
    name: "ai.embedMany",
    attributes: selectTelemetryAttributes({
      telemetry,
      attributes: {
        ...assembleOperationName({ operationName: "ai.embedMany", telemetry }),
        ...baseTelemetryAttributes,
        // specific settings that only make sense on the outer level:
        "ai.values": {
          input: () => values.map((value) => JSON.stringify(value))
        }
      }
    }),
    tracer,
    fn: async (span) => {
      const retry = retryWithExponentialBackoff({ maxRetries });
      const maxEmbeddingsPerCall = model.maxEmbeddingsPerCall;
      if (maxEmbeddingsPerCall == null) {
        const { embeddings: embeddings2, usage } = await retry(() => {
          return recordSpan({
            name: "ai.embedMany.doEmbed",
            attributes: selectTelemetryAttributes({
              telemetry,
              attributes: {
                ...assembleOperationName({
                  operationName: "ai.embedMany.doEmbed",
                  telemetry
                }),
                ...baseTelemetryAttributes,
                // specific settings that only make sense on the outer level:
                "ai.values": {
                  input: () => values.map((value) => JSON.stringify(value))
                }
              }
            }),
            tracer,
            fn: async (doEmbedSpan) => {
              var _a10;
              const modelResponse = await model.doEmbed({
                values,
                abortSignal,
                headers
              });
              const embeddings3 = modelResponse.embeddings;
              const usage2 = (_a10 = modelResponse.usage) != null ? _a10 : { tokens: NaN };
              doEmbedSpan.setAttributes(
                selectTelemetryAttributes({
                  telemetry,
                  attributes: {
                    "ai.embeddings": {
                      output: () => embeddings3.map((embedding) => JSON.stringify(embedding))
                    },
                    "ai.usage.tokens": usage2.tokens
                  }
                })
              );
              return { embeddings: embeddings3, usage: usage2 };
            }
          });
        });
        span.setAttributes(
          selectTelemetryAttributes({
            telemetry,
            attributes: {
              "ai.embeddings": {
                output: () => embeddings2.map((embedding) => JSON.stringify(embedding))
              },
              "ai.usage.tokens": usage.tokens
            }
          })
        );
        return new DefaultEmbedManyResult({ values, embeddings: embeddings2, usage });
      }
      const valueChunks = splitArray(values, maxEmbeddingsPerCall);
      const embeddings = [];
      let tokens = 0;
      for (const chunk of valueChunks) {
        const { embeddings: responseEmbeddings, usage } = await retry(() => {
          return recordSpan({
            name: "ai.embedMany.doEmbed",
            attributes: selectTelemetryAttributes({
              telemetry,
              attributes: {
                ...assembleOperationName({
                  operationName: "ai.embedMany.doEmbed",
                  telemetry
                }),
                ...baseTelemetryAttributes,
                // specific settings that only make sense on the outer level:
                "ai.values": {
                  input: () => chunk.map((value) => JSON.stringify(value))
                }
              }
            }),
            tracer,
            fn: async (doEmbedSpan) => {
              var _a10;
              const modelResponse = await model.doEmbed({
                values: chunk,
                abortSignal,
                headers
              });
              const embeddings2 = modelResponse.embeddings;
              const usage2 = (_a10 = modelResponse.usage) != null ? _a10 : { tokens: NaN };
              doEmbedSpan.setAttributes(
                selectTelemetryAttributes({
                  telemetry,
                  attributes: {
                    "ai.embeddings": {
                      output: () => embeddings2.map((embedding) => JSON.stringify(embedding))
                    },
                    "ai.usage.tokens": usage2.tokens
                  }
                })
              );
              return { embeddings: embeddings2, usage: usage2 };
            }
          });
        });
        embeddings.push(...responseEmbeddings);
        tokens += usage.tokens;
      }
      span.setAttributes(
        selectTelemetryAttributes({
          telemetry,
          attributes: {
            "ai.embeddings": {
              output: () => embeddings.map((embedding) => JSON.stringify(embedding))
            },
            "ai.usage.tokens": tokens
          }
        })
      );
      return new DefaultEmbedManyResult({
        values,
        embeddings,
        usage: { tokens }
      });
    }
  });
}
var DefaultEmbedManyResult = class {
  constructor(options) {
    this.values = options.values;
    this.embeddings = options.embeddings;
    this.usage = options.usage;
  }
};

// core/generate-object/generate-object.ts
var import_provider_utils5 = require("@ai-sdk/provider-utils");

// core/prompt/convert-to-language-model-prompt.ts
var import_provider_utils3 = require("@ai-sdk/provider-utils");

// util/download-error.ts
var import_provider3 = require("@ai-sdk/provider");
var name2 = "AI_DownloadError";
var marker2 = `vercel.ai.error.${name2}`;
var symbol2 = Symbol.for(marker2);
var _a2;
var DownloadError = class extends import_provider3.AISDKError {
  constructor({
    url,
    statusCode,
    statusText,
    cause,
    message = cause == null ? `Failed to download ${url}: ${statusCode} ${statusText}` : `Failed to download ${url}: ${cause}`
  }) {
    super({ name: name2, message, cause });
    this[_a2] = true;
    this.url = url;
    this.statusCode = statusCode;
    this.statusText = statusText;
  }
  static isInstance(error) {
    return import_provider3.AISDKError.hasMarker(error, marker2);
  }
  /**
   * @deprecated use `isInstance` instead
   */
  static isDownloadError(error) {
    return error instanceof Error && error.name === name2 && typeof error.url === "string" && (error.statusCode == null || typeof error.statusCode === "number") && (error.statusText == null || typeof error.statusText === "string");
  }
  /**
   * @deprecated Do not use this method. It will be removed in the next major version.
   */
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      url: this.url,
      statusCode: this.statusCode,
      statusText: this.statusText,
      cause: this.cause
    };
  }
};
_a2 = symbol2;

// util/download.ts
async function download({
  url,
  fetchImplementation = fetch
}) {
  var _a9;
  const urlText = url.toString();
  try {
    const response = await fetchImplementation(urlText);
    if (!response.ok) {
      throw new DownloadError({
        url: urlText,
        statusCode: response.status,
        statusText: response.statusText
      });
    }
    return {
      data: new Uint8Array(await response.arrayBuffer()),
      mimeType: (_a9 = response.headers.get("content-type")) != null ? _a9 : void 0
    };
  } catch (error) {
    if (DownloadError.isInstance(error)) {
      throw error;
    }
    throw new DownloadError({ url: urlText, cause: error });
  }
}

// core/util/detect-image-mimetype.ts
var mimeTypeSignatures = [
  { mimeType: "image/gif", bytes: [71, 73, 70] },
  { mimeType: "image/png", bytes: [137, 80, 78, 71] },
  { mimeType: "image/jpeg", bytes: [255, 216] },
  { mimeType: "image/webp", bytes: [82, 73, 70, 70] }
];
function detectImageMimeType(image) {
  for (const { bytes, mimeType } of mimeTypeSignatures) {
    if (image.length >= bytes.length && bytes.every((byte, index) => image[index] === byte)) {
      return mimeType;
    }
  }
  return void 0;
}

// core/prompt/data-content.ts
var import_provider_utils2 = require("@ai-sdk/provider-utils");

// core/prompt/invalid-data-content-error.ts
var import_provider4 = require("@ai-sdk/provider");
var name3 = "AI_InvalidDataContentError";
var marker3 = `vercel.ai.error.${name3}`;
var symbol3 = Symbol.for(marker3);
var _a3;
var InvalidDataContentError = class extends import_provider4.AISDKError {
  constructor({
    content,
    cause,
    message = `Invalid data content. Expected a base64 string, Uint8Array, ArrayBuffer, or Buffer, but got ${typeof content}.`
  }) {
    super({ name: name3, message, cause });
    this[_a3] = true;
    this.content = content;
  }
  static isInstance(error) {
    return import_provider4.AISDKError.hasMarker(error, marker3);
  }
  /**
   * @deprecated use `isInstance` instead
   */
  static isInvalidDataContentError(error) {
    return error instanceof Error && error.name === name3 && error.content != null;
  }
  /**
   * @deprecated Do not use this method. It will be removed in the next major version.
   */
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      stack: this.stack,
      cause: this.cause,
      content: this.content
    };
  }
};
_a3 = symbol3;

// core/prompt/data-content.ts
function convertDataContentToBase64String(content) {
  if (typeof content === "string") {
    return content;
  }
  if (content instanceof ArrayBuffer) {
    return (0, import_provider_utils2.convertUint8ArrayToBase64)(new Uint8Array(content));
  }
  return (0, import_provider_utils2.convertUint8ArrayToBase64)(content);
}
function convertDataContentToUint8Array(content) {
  if (content instanceof Uint8Array) {
    return content;
  }
  if (typeof content === "string") {
    try {
      return (0, import_provider_utils2.convertBase64ToUint8Array)(content);
    } catch (error) {
      throw new InvalidDataContentError({
        message: "Invalid data content. Content string is not a base64-encoded media.",
        content,
        cause: error
      });
    }
  }
  if (content instanceof ArrayBuffer) {
    return new Uint8Array(content);
  }
  throw new InvalidDataContentError({ content });
}
function convertUint8ArrayToText(uint8Array) {
  try {
    return new TextDecoder().decode(uint8Array);
  } catch (error) {
    throw new Error("Error decoding Uint8Array to text");
  }
}

// core/prompt/invalid-message-role-error.ts
var import_provider5 = require("@ai-sdk/provider");
var name4 = "AI_InvalidMessageRoleError";
var marker4 = `vercel.ai.error.${name4}`;
var symbol4 = Symbol.for(marker4);
var _a4;
var InvalidMessageRoleError = class extends import_provider5.AISDKError {
  constructor({
    role,
    message = `Invalid message role: '${role}'. Must be one of: "system", "user", "assistant", "tool".`
  }) {
    super({ name: name4, message });
    this[_a4] = true;
    this.role = role;
  }
  static isInstance(error) {
    return import_provider5.AISDKError.hasMarker(error, marker4);
  }
  /**
   * @deprecated use `isInstance` instead
   */
  static isInvalidMessageRoleError(error) {
    return error instanceof Error && error.name === name4 && typeof error.role === "string";
  }
  /**
   * @deprecated Do not use this method. It will be removed in the next major version.
   */
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      stack: this.stack,
      role: this.role
    };
  }
};
_a4 = symbol4;

// core/prompt/convert-to-language-model-prompt.ts
async function convertToLanguageModelPrompt({
  prompt,
  modelSupportsImageUrls = true,
  downloadImplementation = download
}) {
  const languageModelMessages = [];
  if (prompt.system != null) {
    languageModelMessages.push({ role: "system", content: prompt.system });
  }
  const downloadedImages = modelSupportsImageUrls || prompt.messages == null ? null : await downloadImages(prompt.messages, downloadImplementation);
  const promptType = prompt.type;
  switch (promptType) {
    case "prompt": {
      languageModelMessages.push({
        role: "user",
        content: [{ type: "text", text: prompt.prompt }]
      });
      break;
    }
    case "messages": {
      languageModelMessages.push(
        ...prompt.messages.map(
          (message) => convertToLanguageModelMessage(message, downloadedImages)
        )
      );
      break;
    }
    default: {
      const _exhaustiveCheck = promptType;
      throw new Error(`Unsupported prompt type: ${_exhaustiveCheck}`);
    }
  }
  return languageModelMessages;
}
function convertToLanguageModelMessage(message, downloadedImages) {
  const role = message.role;
  switch (role) {
    case "system": {
      return { role: "system", content: message.content };
    }
    case "user": {
      if (typeof message.content === "string") {
        return {
          role: "user",
          content: [{ type: "text", text: message.content }]
        };
      }
      return {
        role: "user",
        content: message.content.map(
          (part) => {
            var _a9, _b, _c;
            switch (part.type) {
              case "text": {
                return part;
              }
              case "image": {
                if (part.image instanceof URL) {
                  if (downloadedImages == null) {
                    return {
                      type: "image",
                      image: part.image,
                      mimeType: part.mimeType
                    };
                  } else {
                    const downloadedImage = downloadedImages[part.image.toString()];
                    return {
                      type: "image",
                      image: downloadedImage.data,
                      mimeType: (_a9 = part.mimeType) != null ? _a9 : downloadedImage.mimeType
                    };
                  }
                }
                if (typeof part.image === "string") {
                  try {
                    const url = new URL(part.image);
                    switch (url.protocol) {
                      case "http:":
                      case "https:": {
                        if (downloadedImages == null) {
                          return {
                            type: "image",
                            image: url,
                            mimeType: part.mimeType
                          };
                        } else {
                          const downloadedImage = downloadedImages[part.image];
                          return {
                            type: "image",
                            image: downloadedImage.data,
                            mimeType: (_b = part.mimeType) != null ? _b : downloadedImage.mimeType
                          };
                        }
                      }
                      case "data:": {
                        try {
                          const [header, base64Content] = part.image.split(",");
                          const mimeType = header.split(";")[0].split(":")[1];
                          if (mimeType == null || base64Content == null) {
                            throw new Error("Invalid data URL format");
                          }
                          return {
                            type: "image",
                            image: convertDataContentToUint8Array(base64Content),
                            mimeType
                          };
                        } catch (error) {
                          throw new Error(
                            `Error processing data URL: ${(0, import_provider_utils3.getErrorMessage)(
                              message
                            )}`
                          );
                        }
                      }
                      default: {
                        throw new Error(
                          `Unsupported URL protocol: ${url.protocol}`
                        );
                      }
                    }
                  } catch (_ignored) {
                  }
                }
                const imageUint8 = convertDataContentToUint8Array(part.image);
                return {
                  type: "image",
                  image: imageUint8,
                  mimeType: (_c = part.mimeType) != null ? _c : detectImageMimeType(imageUint8)
                };
              }
            }
          }
        )
      };
    }
    case "assistant": {
      if (typeof message.content === "string") {
        return {
          role: "assistant",
          content: [{ type: "text", text: message.content }]
        };
      }
      return {
        role: "assistant",
        content: message.content.filter(
          // remove empty text parts:
          (part) => part.type !== "text" || part.text !== ""
        )
      };
    }
    case "tool": {
      return message;
    }
    default: {
      const _exhaustiveCheck = role;
      throw new InvalidMessageRoleError({ role: _exhaustiveCheck });
    }
  }
}
async function downloadImages(messages, downloadImplementation) {
  const urls = messages.filter((message) => message.role === "user").map((message) => message.content).filter(
    (content) => Array.isArray(content)
  ).flat().filter((part) => part.type === "image").map((part) => part.image).map(
    (part) => (
      // support string urls in image parts:
      typeof part === "string" && (part.startsWith("http:") || part.startsWith("https:")) ? new URL(part) : part
    )
  ).filter((image) => image instanceof URL);
  const downloadedImages = await Promise.all(
    urls.map(async (url) => ({
      url,
      data: await downloadImplementation({ url })
    }))
  );
  return Object.fromEntries(
    downloadedImages.map(({ url, data }) => [url.toString(), data])
  );
}

// core/prompt/get-validated-prompt.ts
var import_provider6 = require("@ai-sdk/provider");
function getValidatedPrompt(prompt) {
  if (prompt.prompt == null && prompt.messages == null) {
    throw new import_provider6.InvalidPromptError({
      prompt,
      message: "prompt or messages must be defined"
    });
  }
  if (prompt.prompt != null && prompt.messages != null) {
    throw new import_provider6.InvalidPromptError({
      prompt,
      message: "prompt and messages cannot be defined at the same time"
    });
  }
  if (prompt.messages != null) {
    for (const message of prompt.messages) {
      if (message.role === "system" && typeof message.content !== "string") {
        throw new import_provider6.InvalidPromptError({
          prompt,
          message: "system message content must be a string"
        });
      }
    }
  }
  return prompt.prompt != null ? {
    type: "prompt",
    prompt: prompt.prompt,
    messages: void 0,
    system: prompt.system
  } : {
    type: "messages",
    prompt: void 0,
    messages: prompt.messages,
    // only possible case bc of checks above
    system: prompt.system
  };
}

// errors/invalid-argument-error.ts
var import_provider7 = require("@ai-sdk/provider");
var name5 = "AI_InvalidArgumentError";
var marker5 = `vercel.ai.error.${name5}`;
var symbol5 = Symbol.for(marker5);
var _a5;
var InvalidArgumentError = class extends import_provider7.AISDKError {
  constructor({
    parameter,
    value,
    message
  }) {
    super({
      name: name5,
      message: `Invalid argument for parameter ${parameter}: ${message}`
    });
    this[_a5] = true;
    this.parameter = parameter;
    this.value = value;
  }
  static isInstance(error) {
    return import_provider7.AISDKError.hasMarker(error, marker5);
  }
  /**
   * @deprecated use `isInstance` instead
   */
  static isInvalidArgumentError(error) {
    return error instanceof Error && error.name === name5 && typeof error.parameter === "string" && typeof error.value === "string";
  }
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      stack: this.stack,
      parameter: this.parameter,
      value: this.value
    };
  }
};
_a5 = symbol5;

// core/prompt/prepare-call-settings.ts
function prepareCallSettings({
  maxTokens,
  temperature,
  topP,
  presencePenalty,
  frequencyPenalty,
  stopSequences,
  seed,
  maxRetries
}) {
  if (maxTokens != null) {
    if (!Number.isInteger(maxTokens)) {
      throw new InvalidArgumentError({
        parameter: "maxTokens",
        value: maxTokens,
        message: "maxTokens must be an integer"
      });
    }
    if (maxTokens < 1) {
      throw new InvalidArgumentError({
        parameter: "maxTokens",
        value: maxTokens,
        message: "maxTokens must be >= 1"
      });
    }
  }
  if (temperature != null) {
    if (typeof temperature !== "number") {
      throw new InvalidArgumentError({
        parameter: "temperature",
        value: temperature,
        message: "temperature must be a number"
      });
    }
  }
  if (topP != null) {
    if (typeof topP !== "number") {
      throw new InvalidArgumentError({
        parameter: "topP",
        value: topP,
        message: "topP must be a number"
      });
    }
  }
  if (presencePenalty != null) {
    if (typeof presencePenalty !== "number") {
      throw new InvalidArgumentError({
        parameter: "presencePenalty",
        value: presencePenalty,
        message: "presencePenalty must be a number"
      });
    }
  }
  if (frequencyPenalty != null) {
    if (typeof frequencyPenalty !== "number") {
      throw new InvalidArgumentError({
        parameter: "frequencyPenalty",
        value: frequencyPenalty,
        message: "frequencyPenalty must be a number"
      });
    }
  }
  if (seed != null) {
    if (!Number.isInteger(seed)) {
      throw new InvalidArgumentError({
        parameter: "seed",
        value: seed,
        message: "seed must be an integer"
      });
    }
  }
  if (maxRetries != null) {
    if (!Number.isInteger(maxRetries)) {
      throw new InvalidArgumentError({
        parameter: "maxRetries",
        value: maxRetries,
        message: "maxRetries must be an integer"
      });
    }
    if (maxRetries < 0) {
      throw new InvalidArgumentError({
        parameter: "maxRetries",
        value: maxRetries,
        message: "maxRetries must be >= 0"
      });
    }
  }
  return {
    maxTokens,
    temperature: temperature != null ? temperature : 0,
    topP,
    presencePenalty,
    frequencyPenalty,
    stopSequences: stopSequences != null && stopSequences.length > 0 ? stopSequences : void 0,
    seed,
    maxRetries: maxRetries != null ? maxRetries : 2
  };
}

// core/types/token-usage.ts
function calculateCompletionTokenUsage(usage) {
  return {
    promptTokens: usage.promptTokens,
    completionTokens: usage.completionTokens,
    totalTokens: usage.promptTokens + usage.completionTokens
  };
}

// core/util/prepare-response-headers.ts
function prepareResponseHeaders(init, {
  contentType,
  dataStreamVersion
}) {
  var _a9;
  const headers = new Headers((_a9 = init == null ? void 0 : init.headers) != null ? _a9 : {});
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", contentType);
  }
  if (dataStreamVersion !== void 0) {
    headers.set("X-Vercel-AI-Data-Stream", dataStreamVersion);
  }
  return headers;
}

// core/util/schema.ts
var import_provider_utils4 = require("@ai-sdk/provider-utils");
var import_zod_to_json_schema = __toESM(require("zod-to-json-schema"));
var schemaSymbol = Symbol.for("vercel.ai.schema");
function jsonSchema(jsonSchema2, {
  validate
} = {}) {
  return {
    [schemaSymbol]: true,
    _type: void 0,
    // should never be used directly
    [import_provider_utils4.validatorSymbol]: true,
    jsonSchema: jsonSchema2,
    validate
  };
}
function isSchema(value) {
  return typeof value === "object" && value !== null && schemaSymbol in value && value[schemaSymbol] === true && "jsonSchema" in value && "validate" in value;
}
function asSchema(schema) {
  return isSchema(schema) ? schema : zodSchema(schema);
}
function zodSchema(zodSchema2) {
  return jsonSchema(
    // we assume that zodToJsonSchema will return a valid JSONSchema7:
    (0, import_zod_to_json_schema.default)(zodSchema2),
    {
      validate: (value) => {
        const result = zodSchema2.safeParse(value);
        return result.success ? { success: true, value: result.data } : { success: false, error: result.error };
      }
    }
  );
}

// core/generate-object/inject-json-schema-into-system.ts
var DEFAULT_SCHEMA_PREFIX = "JSON schema:";
var DEFAULT_SCHEMA_SUFFIX = "You MUST answer with a JSON object that matches the JSON schema above.";
function injectJsonSchemaIntoSystem({
  system,
  schema,
  schemaPrefix = DEFAULT_SCHEMA_PREFIX,
  schemaSuffix = DEFAULT_SCHEMA_SUFFIX
}) {
  return [
    system,
    system != null ? "" : null,
    // add a newline if system is not null
    schemaPrefix,
    JSON.stringify(schema),
    schemaSuffix
  ].filter((line) => line != null).join("\n");
}

// core/generate-object/no-object-generated-error.ts
var import_provider8 = require("@ai-sdk/provider");
var name6 = "AI_NoObjectGeneratedError";
var marker6 = `vercel.ai.error.${name6}`;
var symbol6 = Symbol.for(marker6);
var _a6;
var NoObjectGeneratedError = class extends import_provider8.AISDKError {
  // used in isInstance
  constructor({ message = "No object generated." } = {}) {
    super({ name: name6, message });
    this[_a6] = true;
  }
  static isInstance(error) {
    return import_provider8.AISDKError.hasMarker(error, marker6);
  }
  /**
   * @deprecated Use isInstance instead.
   */
  static isNoObjectGeneratedError(error) {
    return error instanceof Error && error.name === name6;
  }
  /**
   * @deprecated Do not use this method. It will be removed in the next major version.
   */
  toJSON() {
    return {
      name: this.name,
      cause: this.cause,
      message: this.message,
      stack: this.stack
    };
  }
};
_a6 = symbol6;

// core/generate-object/generate-object.ts
async function generateObject({
  model,
  schema: inputSchema,
  schemaName,
  schemaDescription,
  mode,
  system,
  prompt,
  messages,
  maxRetries,
  abortSignal,
  headers,
  experimental_telemetry: telemetry,
  ...settings
}) {
  var _a9;
  const baseTelemetryAttributes = getBaseTelemetryAttributes({
    model,
    telemetry,
    headers,
    settings: { ...settings, maxRetries }
  });
  const schema = asSchema(inputSchema);
  const tracer = getTracer({ isEnabled: (_a9 = telemetry == null ? void 0 : telemetry.isEnabled) != null ? _a9 : false });
  return recordSpan({
    name: "ai.generateObject",
    attributes: selectTelemetryAttributes({
      telemetry,
      attributes: {
        ...assembleOperationName({
          operationName: "ai.generateObject",
          telemetry
        }),
        ...baseTelemetryAttributes,
        // specific settings that only make sense on the outer level:
        "ai.prompt": {
          input: () => JSON.stringify({ system, prompt, messages })
        },
        "ai.schema": {
          input: () => JSON.stringify(schema.jsonSchema)
        },
        "ai.schema.name": schemaName,
        "ai.schema.description": schemaDescription,
        "ai.settings.mode": mode
      }
    }),
    tracer,
    fn: async (span) => {
      const retry = retryWithExponentialBackoff({ maxRetries });
      if (mode === "auto" || mode == null) {
        mode = model.defaultObjectGenerationMode;
      }
      let result;
      let finishReason;
      let usage;
      let warnings;
      let rawResponse;
      let logprobs;
      switch (mode) {
        case "json": {
          const validatedPrompt = getValidatedPrompt({
            system: model.supportsStructuredOutputs ? system : injectJsonSchemaIntoSystem({
              system,
              schema: schema.jsonSchema
            }),
            prompt,
            messages
          });
          const promptMessages = await convertToLanguageModelPrompt({
            prompt: validatedPrompt,
            modelSupportsImageUrls: model.supportsImageUrls
          });
          const inputFormat = validatedPrompt.type;
          const generateResult = await retry(
            () => recordSpan({
              name: "ai.generateObject.doGenerate",
              attributes: selectTelemetryAttributes({
                telemetry,
                attributes: {
                  ...assembleOperationName({
                    operationName: "ai.generateObject.doGenerate",
                    telemetry
                  }),
                  ...baseTelemetryAttributes,
                  "ai.prompt.format": {
                    input: () => inputFormat
                  },
                  "ai.prompt.messages": {
                    input: () => JSON.stringify(promptMessages)
                  },
                  "ai.settings.mode": mode,
                  // standardized gen-ai llm span attributes:
                  "gen_ai.request.model": model.modelId,
                  "gen_ai.system": model.provider,
                  "gen_ai.request.max_tokens": settings.maxTokens,
                  "gen_ai.request.temperature": settings.temperature,
                  "gen_ai.request.top_p": settings.topP
                }
              }),
              tracer,
              fn: async (span2) => {
                const result2 = await model.doGenerate({
                  mode: {
                    type: "object-json",
                    schema: schema.jsonSchema,
                    name: schemaName,
                    description: schemaDescription
                  },
                  ...prepareCallSettings(settings),
                  inputFormat,
                  prompt: promptMessages,
                  abortSignal,
                  headers
                });
                if (result2.text === void 0) {
                  throw new NoObjectGeneratedError();
                }
                span2.setAttributes(
                  selectTelemetryAttributes({
                    telemetry,
                    attributes: {
                      "ai.finishReason": result2.finishReason,
                      "ai.usage.promptTokens": result2.usage.promptTokens,
                      "ai.usage.completionTokens": result2.usage.completionTokens,
                      "ai.result.object": { output: () => result2.text },
                      // standardized gen-ai llm span attributes:
                      "gen_ai.response.finish_reasons": [result2.finishReason],
                      "gen_ai.usage.prompt_tokens": result2.usage.promptTokens,
                      "gen_ai.usage.completion_tokens": result2.usage.completionTokens
                    }
                  })
                );
                return { ...result2, objectText: result2.text };
              }
            })
          );
          result = generateResult.objectText;
          finishReason = generateResult.finishReason;
          usage = generateResult.usage;
          warnings = generateResult.warnings;
          rawResponse = generateResult.rawResponse;
          logprobs = generateResult.logprobs;
          break;
        }
        case "tool": {
          const validatedPrompt = getValidatedPrompt({
            system,
            prompt,
            messages
          });
          const promptMessages = await convertToLanguageModelPrompt({
            prompt: validatedPrompt,
            modelSupportsImageUrls: model.supportsImageUrls
          });
          const inputFormat = validatedPrompt.type;
          const generateResult = await retry(
            () => recordSpan({
              name: "ai.generateObject.doGenerate",
              attributes: selectTelemetryAttributes({
                telemetry,
                attributes: {
                  ...assembleOperationName({
                    operationName: "ai.generateObject.doGenerate",
                    telemetry
                  }),
                  ...baseTelemetryAttributes,
                  "ai.prompt.format": {
                    input: () => inputFormat
                  },
                  "ai.prompt.messages": {
                    input: () => JSON.stringify(promptMessages)
                  },
                  "ai.settings.mode": mode,
                  // standardized gen-ai llm span attributes:
                  "gen_ai.request.model": model.modelId,
                  "gen_ai.system": model.provider,
                  "gen_ai.request.max_tokens": settings.maxTokens,
                  "gen_ai.request.temperature": settings.temperature,
                  "gen_ai.request.top_p": settings.topP
                }
              }),
              tracer,
              fn: async (span2) => {
                var _a10, _b;
                const result2 = await model.doGenerate({
                  mode: {
                    type: "object-tool",
                    tool: {
                      type: "function",
                      name: schemaName != null ? schemaName : "json",
                      description: schemaDescription != null ? schemaDescription : "Respond with a JSON object.",
                      parameters: schema.jsonSchema
                    }
                  },
                  ...prepareCallSettings(settings),
                  inputFormat,
                  prompt: promptMessages,
                  abortSignal,
                  headers
                });
                const objectText = (_b = (_a10 = result2.toolCalls) == null ? void 0 : _a10[0]) == null ? void 0 : _b.args;
                if (objectText === void 0) {
                  throw new NoObjectGeneratedError();
                }
                span2.setAttributes(
                  selectTelemetryAttributes({
                    telemetry,
                    attributes: {
                      "ai.finishReason": result2.finishReason,
                      "ai.usage.promptTokens": result2.usage.promptTokens,
                      "ai.usage.completionTokens": result2.usage.completionTokens,
                      "ai.result.object": { output: () => objectText },
                      // standardized gen-ai llm span attributes:
                      "gen_ai.response.finish_reasons": [result2.finishReason],
                      "gen_ai.usage.prompt_tokens": result2.usage.promptTokens,
                      "gen_ai.usage.completion_tokens": result2.usage.completionTokens
                    }
                  })
                );
                return { ...result2, objectText };
              }
            })
          );
          result = generateResult.objectText;
          finishReason = generateResult.finishReason;
          usage = generateResult.usage;
          warnings = generateResult.warnings;
          rawResponse = generateResult.rawResponse;
          logprobs = generateResult.logprobs;
          break;
        }
        case void 0: {
          throw new Error(
            "Model does not have a default object generation mode."
          );
        }
        default: {
          const _exhaustiveCheck = mode;
          throw new Error(`Unsupported mode: ${_exhaustiveCheck}`);
        }
      }
      const parseResult = (0, import_provider_utils5.safeParseJSON)({ text: result, schema });
      if (!parseResult.success) {
        throw parseResult.error;
      }
      span.setAttributes(
        selectTelemetryAttributes({
          telemetry,
          attributes: {
            "ai.finishReason": finishReason,
            "ai.usage.promptTokens": usage.promptTokens,
            "ai.usage.completionTokens": usage.completionTokens,
            "ai.result.object": {
              output: () => JSON.stringify(parseResult.value)
            }
          }
        })
      );
      return new DefaultGenerateObjectResult({
        object: parseResult.value,
        finishReason,
        usage: calculateCompletionTokenUsage(usage),
        warnings,
        rawResponse,
        logprobs
      });
    }
  });
}
var DefaultGenerateObjectResult = class {
  constructor(options) {
    this.object = options.object;
    this.finishReason = options.finishReason;
    this.usage = options.usage;
    this.warnings = options.warnings;
    this.rawResponse = options.rawResponse;
    this.logprobs = options.logprobs;
  }
  toJsonResponse(init) {
    var _a9;
    return new Response(JSON.stringify(this.object), {
      status: (_a9 = init == null ? void 0 : init.status) != null ? _a9 : 200,
      headers: prepareResponseHeaders(init, {
        contentType: "application/json; charset=utf-8"
      })
    });
  }
};
var experimental_generateObject = generateObject;

// core/generate-object/stream-object.ts
var import_provider_utils6 = require("@ai-sdk/provider-utils");
var import_ui_utils = require("@ai-sdk/ui-utils");

// util/create-resolvable-promise.ts
function createResolvablePromise() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return {
    promise,
    resolve,
    reject
  };
}

// util/delayed-promise.ts
var DelayedPromise = class {
  constructor() {
    this.status = { type: "pending" };
    this._resolve = void 0;
    this._reject = void 0;
  }
  get value() {
    if (this.promise) {
      return this.promise;
    }
    this.promise = new Promise((resolve, reject) => {
      if (this.status.type === "resolved") {
        resolve(this.status.value);
      } else if (this.status.type === "rejected") {
        reject(this.status.error);
      }
      this._resolve = resolve;
      this._reject = reject;
    });
    return this.promise;
  }
  resolve(value) {
    var _a9;
    this.status = { type: "resolved", value };
    if (this.promise) {
      (_a9 = this._resolve) == null ? void 0 : _a9.call(this, value);
    }
  }
  reject(error) {
    var _a9;
    this.status = { type: "rejected", error };
    if (this.promise) {
      (_a9 = this._reject) == null ? void 0 : _a9.call(this, error);
    }
  }
};

// core/util/async-iterable-stream.ts
function createAsyncIterableStream(source, transformer) {
  const transformedStream = source.pipeThrough(
    new TransformStream(transformer)
  );
  transformedStream[Symbol.asyncIterator] = () => {
    const reader = transformedStream.getReader();
    return {
      async next() {
        const { done, value } = await reader.read();
        return done ? { done: true, value: void 0 } : { done: false, value };
      }
    };
  };
  return transformedStream;
}

// core/generate-object/stream-object.ts
async function streamObject({
  model,
  schema: inputSchema,
  schemaName,
  schemaDescription,
  mode,
  system,
  prompt,
  messages,
  maxRetries,
  abortSignal,
  headers,
  experimental_telemetry: telemetry,
  onFinish,
  ...settings
}) {
  var _a9;
  const baseTelemetryAttributes = getBaseTelemetryAttributes({
    model,
    telemetry,
    headers,
    settings: { ...settings, maxRetries }
  });
  const tracer = getTracer({ isEnabled: (_a9 = telemetry == null ? void 0 : telemetry.isEnabled) != null ? _a9 : false });
  const retry = retryWithExponentialBackoff({ maxRetries });
  const schema = asSchema(inputSchema);
  return recordSpan({
    name: "ai.streamObject",
    attributes: selectTelemetryAttributes({
      telemetry,
      attributes: {
        ...assembleOperationName({
          operationName: "ai.streamObject",
          telemetry
        }),
        ...baseTelemetryAttributes,
        // specific settings that only make sense on the outer level:
        "ai.prompt": {
          input: () => JSON.stringify({ system, prompt, messages })
        },
        "ai.schema": { input: () => JSON.stringify(schema.jsonSchema) },
        "ai.schema.name": schemaName,
        "ai.schema.description": schemaDescription,
        "ai.settings.mode": mode
      }
    }),
    tracer,
    endWhenDone: false,
    fn: async (rootSpan) => {
      if (mode === "auto" || mode == null) {
        mode = model.defaultObjectGenerationMode;
      }
      let callOptions;
      let transformer;
      switch (mode) {
        case "json": {
          const validatedPrompt = getValidatedPrompt({
            system: model.supportsStructuredOutputs ? system : injectJsonSchemaIntoSystem({
              system,
              schema: schema.jsonSchema
            }),
            prompt,
            messages
          });
          callOptions = {
            mode: {
              type: "object-json",
              schema: schema.jsonSchema,
              name: schemaName,
              description: schemaDescription
            },
            ...prepareCallSettings(settings),
            inputFormat: validatedPrompt.type,
            prompt: await convertToLanguageModelPrompt({
              prompt: validatedPrompt,
              modelSupportsImageUrls: model.supportsImageUrls
            }),
            abortSignal,
            headers
          };
          transformer = {
            transform: (chunk, controller) => {
              switch (chunk.type) {
                case "text-delta":
                  controller.enqueue(chunk.textDelta);
                  break;
                case "finish":
                case "error":
                  controller.enqueue(chunk);
                  break;
              }
            }
          };
          break;
        }
        case "tool": {
          const validatedPrompt = getValidatedPrompt({
            system,
            prompt,
            messages
          });
          callOptions = {
            mode: {
              type: "object-tool",
              tool: {
                type: "function",
                name: schemaName != null ? schemaName : "json",
                description: schemaDescription != null ? schemaDescription : "Respond with a JSON object.",
                parameters: schema.jsonSchema
              }
            },
            ...prepareCallSettings(settings),
            inputFormat: validatedPrompt.type,
            prompt: await convertToLanguageModelPrompt({
              prompt: validatedPrompt,
              modelSupportsImageUrls: model.supportsImageUrls
            }),
            abortSignal,
            headers
          };
          transformer = {
            transform(chunk, controller) {
              switch (chunk.type) {
                case "tool-call-delta":
                  controller.enqueue(chunk.argsTextDelta);
                  break;
                case "finish":
                case "error":
                  controller.enqueue(chunk);
                  break;
              }
            }
          };
          break;
        }
        case void 0: {
          throw new Error(
            "Model does not have a default object generation mode."
          );
        }
        default: {
          const _exhaustiveCheck = mode;
          throw new Error(`Unsupported mode: ${_exhaustiveCheck}`);
        }
      }
      const {
        result: { stream, warnings, rawResponse },
        doStreamSpan
      } = await retry(
        () => recordSpan({
          name: "ai.streamObject.doStream",
          attributes: selectTelemetryAttributes({
            telemetry,
            attributes: {
              ...assembleOperationName({
                operationName: "ai.streamObject.doStream",
                telemetry
              }),
              ...baseTelemetryAttributes,
              "ai.prompt.format": {
                input: () => callOptions.inputFormat
              },
              "ai.prompt.messages": {
                input: () => JSON.stringify(callOptions.prompt)
              },
              "ai.settings.mode": mode,
              // standardized gen-ai llm span attributes:
              "gen_ai.request.model": model.modelId,
              "gen_ai.system": model.provider,
              "gen_ai.request.max_tokens": settings.maxTokens,
              "gen_ai.request.temperature": settings.temperature,
              "gen_ai.request.top_p": settings.topP
            }
          }),
          tracer,
          endWhenDone: false,
          fn: async (doStreamSpan2) => ({
            result: await model.doStream(callOptions),
            doStreamSpan: doStreamSpan2
          })
        })
      );
      return new DefaultStreamObjectResult({
        stream: stream.pipeThrough(new TransformStream(transformer)),
        warnings,
        rawResponse,
        schema,
        onFinish,
        rootSpan,
        doStreamSpan,
        telemetry
      });
    }
  });
}
var DefaultStreamObjectResult = class {
  constructor({
    stream,
    warnings,
    rawResponse,
    schema,
    onFinish,
    rootSpan,
    doStreamSpan,
    telemetry
  }) {
    this.warnings = warnings;
    this.rawResponse = rawResponse;
    this.objectPromise = new DelayedPromise();
    const { resolve: resolveUsage, promise: usagePromise } = createResolvablePromise();
    this.usage = usagePromise;
    let usage;
    let finishReason;
    let object;
    let error;
    let accumulatedText = "";
    let delta = "";
    let latestObject = void 0;
    let firstChunk = true;
    const self = this;
    this.originalStream = stream.pipeThrough(
      new TransformStream({
        async transform(chunk, controller) {
          if (firstChunk) {
            firstChunk = false;
            doStreamSpan.addEvent("ai.stream.firstChunk");
          }
          if (typeof chunk === "string") {
            accumulatedText += chunk;
            delta += chunk;
            const currentObject = (0, import_ui_utils.parsePartialJson)(
              accumulatedText
            );
            if (!(0, import_ui_utils.isDeepEqualData)(latestObject, currentObject)) {
              latestObject = currentObject;
              controller.enqueue({
                type: "object",
                object: currentObject
              });
              controller.enqueue({
                type: "text-delta",
                textDelta: delta
              });
              delta = "";
            }
            return;
          }
          switch (chunk.type) {
            case "finish": {
              if (delta !== "") {
                controller.enqueue({
                  type: "text-delta",
                  textDelta: delta
                });
              }
              finishReason = chunk.finishReason;
              usage = calculateCompletionTokenUsage(chunk.usage);
              controller.enqueue({ ...chunk, usage });
              resolveUsage(usage);
              const validationResult = (0, import_provider_utils6.safeValidateTypes)({
                value: latestObject,
                schema
              });
              if (validationResult.success) {
                object = validationResult.value;
                self.objectPromise.resolve(object);
              } else {
                error = validationResult.error;
                self.objectPromise.reject(error);
              }
              break;
            }
            default: {
              controller.enqueue(chunk);
              break;
            }
          }
        },
        // invoke onFinish callback and resolve toolResults promise when the stream is about to close:
        async flush(controller) {
          try {
            const finalUsage = usage != null ? usage : {
              promptTokens: NaN,
              completionTokens: NaN,
              totalTokens: NaN
            };
            doStreamSpan.setAttributes(
              selectTelemetryAttributes({
                telemetry,
                attributes: {
                  "ai.finishReason": finishReason,
                  "ai.usage.promptTokens": finalUsage.promptTokens,
                  "ai.usage.completionTokens": finalUsage.completionTokens,
                  "ai.result.object": {
                    output: () => JSON.stringify(object)
                  },
                  // standardized gen-ai llm span attributes:
                  "gen_ai.usage.prompt_tokens": finalUsage.promptTokens,
                  "gen_ai.usage.completion_tokens": finalUsage.completionTokens,
                  "gen_ai.response.finish_reasons": [finishReason]
                }
              })
            );
            doStreamSpan.end();
            rootSpan.setAttributes(
              selectTelemetryAttributes({
                telemetry,
                attributes: {
                  "ai.usage.promptTokens": finalUsage.promptTokens,
                  "ai.usage.completionTokens": finalUsage.completionTokens,
                  "ai.result.object": {
                    output: () => JSON.stringify(object)
                  }
                }
              })
            );
            await (onFinish == null ? void 0 : onFinish({
              usage: finalUsage,
              object,
              error,
              rawResponse,
              warnings
            }));
          } catch (error2) {
            controller.error(error2);
          } finally {
            rootSpan.end();
          }
        }
      })
    );
  }
  get object() {
    return this.objectPromise.value;
  }
  get partialObjectStream() {
    return createAsyncIterableStream(this.originalStream, {
      transform(chunk, controller) {
        switch (chunk.type) {
          case "object":
            controller.enqueue(chunk.object);
            break;
          case "text-delta":
          case "finish":
            break;
          case "error":
            controller.error(chunk.error);
            break;
          default: {
            const _exhaustiveCheck = chunk;
            throw new Error(`Unsupported chunk type: ${_exhaustiveCheck}`);
          }
        }
      }
    });
  }
  get textStream() {
    return createAsyncIterableStream(this.originalStream, {
      transform(chunk, controller) {
        switch (chunk.type) {
          case "text-delta":
            controller.enqueue(chunk.textDelta);
            break;
          case "object":
          case "finish":
            break;
          case "error":
            controller.error(chunk.error);
            break;
          default: {
            const _exhaustiveCheck = chunk;
            throw new Error(`Unsupported chunk type: ${_exhaustiveCheck}`);
          }
        }
      }
    });
  }
  get fullStream() {
    return createAsyncIterableStream(this.originalStream, {
      transform(chunk, controller) {
        controller.enqueue(chunk);
      }
    });
  }
  pipeTextStreamToResponse(response, init) {
    var _a9;
    response.writeHead((_a9 = init == null ? void 0 : init.status) != null ? _a9 : 200, {
      "Content-Type": "text/plain; charset=utf-8",
      ...init == null ? void 0 : init.headers
    });
    const reader = this.textStream.pipeThrough(new TextEncoderStream()).getReader();
    const read = async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done)
            break;
          response.write(value);
        }
      } catch (error) {
        throw error;
      } finally {
        response.end();
      }
    };
    read();
  }
  toTextStreamResponse(init) {
    var _a9;
    return new Response(this.textStream.pipeThrough(new TextEncoderStream()), {
      status: (_a9 = init == null ? void 0 : init.status) != null ? _a9 : 200,
      headers: prepareResponseHeaders(init, {
        contentType: "text/plain; charset=utf-8"
      })
    });
  }
};
var experimental_streamObject = streamObject;

// core/util/is-non-empty-object.ts
function isNonEmptyObject(object) {
  return object != null && Object.keys(object).length > 0;
}

// core/prompt/prepare-tools-and-tool-choice.ts
function prepareToolsAndToolChoice({
  tools,
  toolChoice
}) {
  if (!isNonEmptyObject(tools)) {
    return {
      tools: void 0,
      toolChoice: void 0
    };
  }
  return {
    tools: Object.entries(tools).map(([name9, tool2]) => ({
      type: "function",
      name: name9,
      description: tool2.description,
      parameters: asSchema(tool2.parameters).jsonSchema
    })),
    toolChoice: toolChoice == null ? { type: "auto" } : typeof toolChoice === "string" ? { type: toolChoice } : { type: "tool", toolName: toolChoice.toolName }
  };
}

// core/generate-text/tool-call.ts
var import_provider_utils7 = require("@ai-sdk/provider-utils");

// errors/invalid-tool-arguments-error.ts
var import_provider9 = require("@ai-sdk/provider");
var name7 = "AI_InvalidToolArgumentsError";
var marker7 = `vercel.ai.error.${name7}`;
var symbol7 = Symbol.for(marker7);
var _a7;
var InvalidToolArgumentsError = class extends import_provider9.AISDKError {
  constructor({
    toolArgs,
    toolName,
    cause,
    message = `Invalid arguments for tool ${toolName}: ${(0, import_provider9.getErrorMessage)(
      cause
    )}`
  }) {
    super({ name: name7, message, cause });
    this[_a7] = true;
    this.toolArgs = toolArgs;
    this.toolName = toolName;
  }
  static isInstance(error) {
    return import_provider9.AISDKError.hasMarker(error, marker7);
  }
  /**
   * @deprecated use `isInstance` instead
   */
  static isInvalidToolArgumentsError(error) {
    return error instanceof Error && error.name === name7 && typeof error.toolName === "string" && typeof error.toolArgs === "string";
  }
  /**
   * @deprecated Do not use this method. It will be removed in the next major version.
   */
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      cause: this.cause,
      stack: this.stack,
      toolName: this.toolName,
      toolArgs: this.toolArgs
    };
  }
};
_a7 = symbol7;

// errors/no-such-tool-error.ts
var import_provider10 = require("@ai-sdk/provider");
var name8 = "AI_NoSuchToolError";
var marker8 = `vercel.ai.error.${name8}`;
var symbol8 = Symbol.for(marker8);
var _a8;
var NoSuchToolError = class extends import_provider10.AISDKError {
  constructor({
    toolName,
    availableTools = void 0,
    message = `Model tried to call unavailable tool '${toolName}'. ${availableTools === void 0 ? "No tools are available." : `Available tools: ${availableTools.join(", ")}.`}`
  }) {
    super({ name: name8, message });
    this[_a8] = true;
    this.toolName = toolName;
    this.availableTools = availableTools;
  }
  static isInstance(error) {
    return import_provider10.AISDKError.hasMarker(error, marker8);
  }
  /**
   * @deprecated use `isInstance` instead
   */
  static isNoSuchToolError(error) {
    return error instanceof Error && error.name === name8 && "toolName" in error && error.toolName != void 0 && typeof error.name === "string";
  }
  /**
   * @deprecated Do not use this method. It will be removed in the next major version.
   */
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      stack: this.stack,
      toolName: this.toolName,
      availableTools: this.availableTools
    };
  }
};
_a8 = symbol8;

// core/generate-text/tool-call.ts
function parseToolCall({
  toolCall,
  tools
}) {
  const toolName = toolCall.toolName;
  if (tools == null) {
    throw new NoSuchToolError({ toolName: toolCall.toolName });
  }
  const tool2 = tools[toolName];
  if (tool2 == null) {
    throw new NoSuchToolError({
      toolName: toolCall.toolName,
      availableTools: Object.keys(tools)
    });
  }
  const parseResult = (0, import_provider_utils7.safeParseJSON)({
    text: toolCall.args,
    schema: asSchema(tool2.parameters)
  });
  if (parseResult.success === false) {
    throw new InvalidToolArgumentsError({
      toolName,
      toolArgs: toolCall.args,
      cause: parseResult.error
    });
  }
  return {
    type: "tool-call",
    toolCallId: toolCall.toolCallId,
    toolName,
    args: parseResult.value
  };
}

// core/generate-text/generate-text.ts
async function generateText({
  model,
  tools,
  toolChoice,
  system,
  prompt,
  messages,
  maxRetries,
  abortSignal,
  headers,
  maxAutomaticRoundtrips = 0,
  maxToolRoundtrips = maxAutomaticRoundtrips,
  experimental_telemetry: telemetry,
  ...settings
}) {
  var _a9;
  const baseTelemetryAttributes = getBaseTelemetryAttributes({
    model,
    telemetry,
    headers,
    settings: { ...settings, maxRetries }
  });
  const tracer = getTracer({ isEnabled: (_a9 = telemetry == null ? void 0 : telemetry.isEnabled) != null ? _a9 : false });
  return recordSpan({
    name: "ai.generateText",
    attributes: selectTelemetryAttributes({
      telemetry,
      attributes: {
        ...assembleOperationName({
          operationName: "ai.generateText",
          telemetry
        }),
        ...baseTelemetryAttributes,
        // specific settings that only make sense on the outer level:
        "ai.prompt": {
          input: () => JSON.stringify({ system, prompt, messages })
        },
        "ai.settings.maxToolRoundtrips": maxToolRoundtrips
      }
    }),
    tracer,
    fn: async (span) => {
      var _a10, _b, _c, _d;
      const retry = retryWithExponentialBackoff({ maxRetries });
      const validatedPrompt = getValidatedPrompt({
        system,
        prompt,
        messages
      });
      const mode = {
        type: "regular",
        ...prepareToolsAndToolChoice({ tools, toolChoice })
      };
      const callSettings = prepareCallSettings(settings);
      const promptMessages = await convertToLanguageModelPrompt({
        prompt: validatedPrompt,
        modelSupportsImageUrls: model.supportsImageUrls
      });
      let currentModelResponse;
      let currentToolCalls = [];
      let currentToolResults = [];
      let roundtripCount = 0;
      const responseMessages = [];
      const roundtrips = [];
      const usage = {
        completionTokens: 0,
        promptTokens: 0,
        totalTokens: 0
      };
      do {
        const currentInputFormat = roundtripCount === 0 ? validatedPrompt.type : "messages";
        currentModelResponse = await retry(
          () => recordSpan({
            name: "ai.generateText.doGenerate",
            attributes: selectTelemetryAttributes({
              telemetry,
              attributes: {
                ...assembleOperationName({
                  operationName: "ai.generateText.doGenerate",
                  telemetry
                }),
                ...baseTelemetryAttributes,
                "ai.prompt.format": { input: () => currentInputFormat },
                "ai.prompt.messages": {
                  input: () => JSON.stringify(promptMessages)
                },
                // standardized gen-ai llm span attributes:
                "gen_ai.request.model": model.modelId,
                "gen_ai.system": model.provider,
                "gen_ai.request.max_tokens": settings.maxTokens,
                "gen_ai.request.temperature": settings.temperature,
                "gen_ai.request.top_p": settings.topP
              }
            }),
            tracer,
            fn: async (span2) => {
              const result = await model.doGenerate({
                mode,
                ...callSettings,
                inputFormat: currentInputFormat,
                prompt: promptMessages,
                abortSignal,
                headers
              });
              span2.setAttributes(
                selectTelemetryAttributes({
                  telemetry,
                  attributes: {
                    "ai.finishReason": result.finishReason,
                    "ai.usage.promptTokens": result.usage.promptTokens,
                    "ai.usage.completionTokens": result.usage.completionTokens,
                    "ai.result.text": {
                      output: () => result.text
                    },
                    "ai.result.toolCalls": {
                      output: () => JSON.stringify(result.toolCalls)
                    },
                    // standardized gen-ai llm span attributes:
                    "gen_ai.response.finish_reasons": [result.finishReason],
                    "gen_ai.usage.prompt_tokens": result.usage.promptTokens,
                    "gen_ai.usage.completion_tokens": result.usage.completionTokens
                  }
                })
              );
              return result;
            }
          })
        );
        currentToolCalls = ((_a10 = currentModelResponse.toolCalls) != null ? _a10 : []).map(
          (modelToolCall) => parseToolCall({ toolCall: modelToolCall, tools })
        );
        currentToolResults = tools == null ? [] : await executeTools({
          toolCalls: currentToolCalls,
          tools,
          tracer,
          telemetry
        });
        const currentUsage = calculateCompletionTokenUsage(
          currentModelResponse.usage
        );
        usage.completionTokens += currentUsage.completionTokens;
        usage.promptTokens += currentUsage.promptTokens;
        usage.totalTokens += currentUsage.totalTokens;
        roundtrips.push({
          text: (_b = currentModelResponse.text) != null ? _b : "",
          toolCalls: currentToolCalls,
          toolResults: currentToolResults,
          finishReason: currentModelResponse.finishReason,
          usage: currentUsage,
          warnings: currentModelResponse.warnings,
          logprobs: currentModelResponse.logprobs
        });
        const newResponseMessages = toResponseMessages({
          text: (_c = currentModelResponse.text) != null ? _c : "",
          toolCalls: currentToolCalls,
          toolResults: currentToolResults
        });
        responseMessages.push(...newResponseMessages);
        promptMessages.push(
          ...newResponseMessages.map(
            (message) => convertToLanguageModelMessage(message, null)
          )
        );
      } while (
        // there are tool calls:
        currentToolCalls.length > 0 && // all current tool calls have results:
        currentToolResults.length === currentToolCalls.length && // the number of roundtrips is less than the maximum:
        roundtripCount++ < maxToolRoundtrips
      );
      span.setAttributes(
        selectTelemetryAttributes({
          telemetry,
          attributes: {
            "ai.finishReason": currentModelResponse.finishReason,
            "ai.usage.promptTokens": currentModelResponse.usage.promptTokens,
            "ai.usage.completionTokens": currentModelResponse.usage.completionTokens,
            "ai.result.text": {
              output: () => currentModelResponse.text
            },
            "ai.result.toolCalls": {
              output: () => JSON.stringify(currentModelResponse.toolCalls)
            }
          }
        })
      );
      return new DefaultGenerateTextResult({
        // Always return a string so that the caller doesn't have to check for undefined.
        // If they need to check if the model did not return any text,
        // they can check the length of the string:
        text: (_d = currentModelResponse.text) != null ? _d : "",
        toolCalls: currentToolCalls,
        toolResults: currentToolResults,
        finishReason: currentModelResponse.finishReason,
        usage,
        warnings: currentModelResponse.warnings,
        rawResponse: currentModelResponse.rawResponse,
        logprobs: currentModelResponse.logprobs,
        responseMessages,
        roundtrips
      });
    }
  });
}
async function executeTools({
  toolCalls,
  tools,
  tracer,
  telemetry
}) {
  const toolResults = await Promise.all(
    toolCalls.map(async (toolCall) => {
      const tool2 = tools[toolCall.toolName];
      if ((tool2 == null ? void 0 : tool2.execute) == null) {
        return void 0;
      }
      const result = await recordSpan({
        name: "ai.toolCall",
        attributes: selectTelemetryAttributes({
          telemetry,
          attributes: {
            ...assembleOperationName({
              operationName: "ai.toolCall",
              telemetry
            }),
            "ai.toolCall.name": toolCall.toolName,
            "ai.toolCall.id": toolCall.toolCallId,
            "ai.toolCall.args": {
              output: () => JSON.stringify(toolCall.args)
            }
          }
        }),
        tracer,
        fn: async (span) => {
          const result2 = await tool2.execute(toolCall.args);
          try {
            span.setAttributes(
              selectTelemetryAttributes({
                telemetry,
                attributes: {
                  "ai.toolCall.result": {
                    output: () => JSON.stringify(result2)
                  }
                }
              })
            );
          } catch (ignored) {
          }
          return result2;
        }
      });
      return {
        toolCallId: toolCall.toolCallId,
        toolName: toolCall.toolName,
        args: toolCall.args,
        result
      };
    })
  );
  return toolResults.filter(
    (result) => result != null
  );
}
var DefaultGenerateTextResult = class {
  constructor(options) {
    this.text = options.text;
    this.toolCalls = options.toolCalls;
    this.toolResults = options.toolResults;
    this.finishReason = options.finishReason;
    this.usage = options.usage;
    this.warnings = options.warnings;
    this.rawResponse = options.rawResponse;
    this.logprobs = options.logprobs;
    this.responseMessages = options.responseMessages;
    this.roundtrips = options.roundtrips;
  }
};
function toResponseMessages({
  text,
  toolCalls,
  toolResults
}) {
  const responseMessages = [];
  responseMessages.push({
    role: "assistant",
    content: [{ type: "text", text }, ...toolCalls]
  });
  if (toolResults.length > 0) {
    responseMessages.push({
      role: "tool",
      content: toolResults.map((result) => ({
        type: "tool-result",
        toolCallId: result.toolCallId,
        toolName: result.toolName,
        result: result.result
      }))
    });
  }
  return responseMessages;
}
var experimental_generateText = generateText;

// core/util/merge-streams.ts
function mergeStreams(stream1, stream2) {
  const reader1 = stream1.getReader();
  const reader2 = stream2.getReader();
  let lastRead1 = void 0;
  let lastRead2 = void 0;
  let stream1Done = false;
  let stream2Done = false;
  async function readStream1(controller) {
    try {
      if (lastRead1 == null) {
        lastRead1 = reader1.read();
      }
      const result = await lastRead1;
      lastRead1 = void 0;
      if (!result.done) {
        controller.enqueue(result.value);
      } else {
        controller.close();
      }
    } catch (error) {
      controller.error(error);
    }
  }
  async function readStream2(controller) {
    try {
      if (lastRead2 == null) {
        lastRead2 = reader2.read();
      }
      const result = await lastRead2;
      lastRead2 = void 0;
      if (!result.done) {
        controller.enqueue(result.value);
      } else {
        controller.close();
      }
    } catch (error) {
      controller.error(error);
    }
  }
  return new ReadableStream({
    async pull(controller) {
      try {
        if (stream1Done) {
          await readStream2(controller);
          return;
        }
        if (stream2Done) {
          await readStream1(controller);
          return;
        }
        if (lastRead1 == null) {
          lastRead1 = reader1.read();
        }
        if (lastRead2 == null) {
          lastRead2 = reader2.read();
        }
        const { result, reader } = await Promise.race([
          lastRead1.then((result2) => ({ result: result2, reader: reader1 })),
          lastRead2.then((result2) => ({ result: result2, reader: reader2 }))
        ]);
        if (!result.done) {
          controller.enqueue(result.value);
        }
        if (reader === reader1) {
          lastRead1 = void 0;
          if (result.done) {
            await readStream2(controller);
            stream1Done = true;
          }
        } else {
          lastRead2 = void 0;
          if (result.done) {
            stream2Done = true;
            await readStream1(controller);
          }
        }
      } catch (error) {
        controller.error(error);
      }
    },
    cancel() {
      reader1.cancel();
      reader2.cancel();
    }
  });
}

// core/generate-text/run-tools-transformation.ts
var import_ui_utils2 = require("@ai-sdk/ui-utils");
function runToolsTransformation({
  tools,
  generatorStream,
  toolCallStreaming,
  tracer,
  telemetry
}) {
  let canClose = false;
  const outstandingToolCalls = /* @__PURE__ */ new Set();
  let toolResultsStreamController = null;
  const toolResultsStream = new ReadableStream({
    start(controller) {
      toolResultsStreamController = controller;
    }
  });
  const activeToolCalls = {};
  const forwardStream = new TransformStream({
    transform(chunk, controller) {
      const chunkType = chunk.type;
      switch (chunkType) {
        case "text-delta":
        case "error": {
          controller.enqueue(chunk);
          break;
        }
        case "tool-call-delta": {
          if (toolCallStreaming) {
            if (!activeToolCalls[chunk.toolCallId]) {
              controller.enqueue({
                type: "tool-call-streaming-start",
                toolCallId: chunk.toolCallId,
                toolName: chunk.toolName
              });
              activeToolCalls[chunk.toolCallId] = true;
            }
            controller.enqueue({
              type: "tool-call-delta",
              toolCallId: chunk.toolCallId,
              toolName: chunk.toolName,
              argsTextDelta: chunk.argsTextDelta
            });
          }
          break;
        }
        case "tool-call": {
          const toolName = chunk.toolName;
          if (tools == null) {
            toolResultsStreamController.enqueue({
              type: "error",
              error: new NoSuchToolError({ toolName: chunk.toolName })
            });
            break;
          }
          const tool2 = tools[toolName];
          if (tool2 == null) {
            toolResultsStreamController.enqueue({
              type: "error",
              error: new NoSuchToolError({
                toolName: chunk.toolName,
                availableTools: Object.keys(tools)
              })
            });
            break;
          }
          try {
            const toolCall = parseToolCall({
              toolCall: chunk,
              tools
            });
            controller.enqueue(toolCall);
            if (tool2.execute != null) {
              const toolExecutionId = (0, import_ui_utils2.generateId)();
              outstandingToolCalls.add(toolExecutionId);
              recordSpan({
                name: "ai.toolCall",
                attributes: selectTelemetryAttributes({
                  telemetry,
                  attributes: {
                    ...assembleOperationName({
                      operationName: "ai.toolCall",
                      telemetry
                    }),
                    "ai.toolCall.name": toolCall.toolName,
                    "ai.toolCall.id": toolCall.toolCallId,
                    "ai.toolCall.args": {
                      output: () => JSON.stringify(toolCall.args)
                    }
                  }
                }),
                tracer,
                fn: async (span) => tool2.execute(toolCall.args).then(
                  (result) => {
                    toolResultsStreamController.enqueue({
                      ...toolCall,
                      type: "tool-result",
                      result
                    });
                    outstandingToolCalls.delete(toolExecutionId);
                    if (canClose && outstandingToolCalls.size === 0) {
                      toolResultsStreamController.close();
                    }
                    try {
                      span.setAttributes(
                        selectTelemetryAttributes({
                          telemetry,
                          attributes: {
                            "ai.toolCall.result": {
                              output: () => JSON.stringify(result)
                            }
                          }
                        })
                      );
                    } catch (ignored) {
                    }
                  },
                  (error) => {
                    toolResultsStreamController.enqueue({
                      type: "error",
                      error
                    });
                    outstandingToolCalls.delete(toolExecutionId);
                    if (canClose && outstandingToolCalls.size === 0) {
                      toolResultsStreamController.close();
                    }
                  }
                )
              });
            }
          } catch (error) {
            toolResultsStreamController.enqueue({
              type: "error",
              error
            });
          }
          break;
        }
        case "finish": {
          controller.enqueue({
            type: "finish",
            finishReason: chunk.finishReason,
            logprobs: chunk.logprobs,
            usage: calculateCompletionTokenUsage(chunk.usage)
          });
          break;
        }
        default: {
          const _exhaustiveCheck = chunkType;
          throw new Error(`Unhandled chunk type: ${_exhaustiveCheck}`);
        }
      }
    },
    flush() {
      canClose = true;
      if (outstandingToolCalls.size === 0) {
        toolResultsStreamController.close();
      }
    }
  });
  return new ReadableStream({
    async start(controller) {
      return Promise.all([
        generatorStream.pipeThrough(forwardStream).pipeTo(
          new WritableStream({
            write(chunk) {
              controller.enqueue(chunk);
            },
            close() {
            }
          })
        ),
        toolResultsStream.pipeTo(
          new WritableStream({
            write(chunk) {
              controller.enqueue(chunk);
            },
            close() {
              controller.close();
            }
          })
        )
      ]);
    }
  });
}

// core/generate-text/stream-text.ts
async function streamText({
  model,
  tools,
  toolChoice,
  system,
  prompt,
  messages,
  maxRetries,
  abortSignal,
  headers,
  experimental_telemetry: telemetry,
  experimental_toolCallStreaming: toolCallStreaming = false,
  onFinish,
  ...settings
}) {
  var _a9;
  const baseTelemetryAttributes = getBaseTelemetryAttributes({
    model,
    telemetry,
    headers,
    settings: { ...settings, maxRetries }
  });
  const tracer = getTracer({ isEnabled: (_a9 = telemetry == null ? void 0 : telemetry.isEnabled) != null ? _a9 : false });
  return recordSpan({
    name: "ai.streamText",
    attributes: selectTelemetryAttributes({
      telemetry,
      attributes: {
        ...assembleOperationName({ operationName: "ai.streamText", telemetry }),
        ...baseTelemetryAttributes,
        // specific settings that only make sense on the outer level:
        "ai.prompt": {
          input: () => JSON.stringify({ system, prompt, messages })
        }
      }
    }),
    tracer,
    endWhenDone: false,
    fn: async (rootSpan) => {
      const retry = retryWithExponentialBackoff({ maxRetries });
      const validatedPrompt = getValidatedPrompt({ system, prompt, messages });
      const promptMessages = await convertToLanguageModelPrompt({
        prompt: validatedPrompt,
        modelSupportsImageUrls: model.supportsImageUrls
      });
      const {
        result: { stream, warnings, rawResponse },
        doStreamSpan
      } = await retry(
        () => recordSpan({
          name: "ai.streamText.doStream",
          attributes: selectTelemetryAttributes({
            telemetry,
            attributes: {
              ...assembleOperationName({
                operationName: "ai.streamText.doStream",
                telemetry
              }),
              ...baseTelemetryAttributes,
              "ai.prompt.format": {
                input: () => validatedPrompt.type
              },
              "ai.prompt.messages": {
                input: () => JSON.stringify(promptMessages)
              },
              // standardized gen-ai llm span attributes:
              "gen_ai.request.model": model.modelId,
              "gen_ai.system": model.provider,
              "gen_ai.request.max_tokens": settings.maxTokens,
              "gen_ai.request.temperature": settings.temperature,
              "gen_ai.request.top_p": settings.topP
            }
          }),
          tracer,
          endWhenDone: false,
          fn: async (doStreamSpan2) => {
            return {
              result: await model.doStream({
                mode: {
                  type: "regular",
                  ...prepareToolsAndToolChoice({ tools, toolChoice })
                },
                ...prepareCallSettings(settings),
                inputFormat: validatedPrompt.type,
                prompt: promptMessages,
                abortSignal,
                headers
              }),
              doStreamSpan: doStreamSpan2
            };
          }
        })
      );
      return new DefaultStreamTextResult({
        stream: runToolsTransformation({
          tools,
          generatorStream: stream,
          toolCallStreaming,
          tracer,
          telemetry
        }),
        warnings,
        rawResponse,
        onFinish,
        rootSpan,
        doStreamSpan,
        telemetry
      });
    }
  });
}
var DefaultStreamTextResult = class {
  constructor({
    stream,
    warnings,
    rawResponse,
    onFinish,
    rootSpan,
    doStreamSpan,
    telemetry
  }) {
    this.warnings = warnings;
    this.rawResponse = rawResponse;
    this.onFinish = onFinish;
    const { resolve: resolveUsage, promise: usagePromise } = createResolvablePromise();
    this.usage = usagePromise;
    const { resolve: resolveFinishReason, promise: finishReasonPromise } = createResolvablePromise();
    this.finishReason = finishReasonPromise;
    const { resolve: resolveText, promise: textPromise } = createResolvablePromise();
    this.text = textPromise;
    const { resolve: resolveToolCalls, promise: toolCallsPromise } = createResolvablePromise();
    this.toolCalls = toolCallsPromise;
    const { resolve: resolveToolResults, promise: toolResultsPromise } = createResolvablePromise();
    this.toolResults = toolResultsPromise;
    let finishReason;
    let usage;
    let text = "";
    const toolCalls = [];
    const toolResults = [];
    let firstChunk = true;
    const self = this;
    this.originalStream = stream.pipeThrough(
      new TransformStream({
        async transform(chunk, controller) {
          controller.enqueue(chunk);
          if (firstChunk) {
            firstChunk = false;
            doStreamSpan.addEvent("ai.stream.firstChunk");
          }
          const chunkType = chunk.type;
          switch (chunkType) {
            case "text-delta":
              text += chunk.textDelta;
              break;
            case "tool-call":
              toolCalls.push(chunk);
              break;
            case "tool-result":
              toolResults.push(chunk);
              break;
            case "finish":
              usage = chunk.usage;
              finishReason = chunk.finishReason;
              resolveUsage(usage);
              resolveFinishReason(finishReason);
              resolveText(text);
              resolveToolCalls(toolCalls);
              break;
            case "tool-call-streaming-start":
            case "tool-call-delta":
            case "error":
              break;
            default: {
              const exhaustiveCheck = chunkType;
              throw new Error(`Unknown chunk type: ${exhaustiveCheck}`);
            }
          }
        },
        // invoke onFinish callback and resolve toolResults promise when the stream is about to close:
        async flush(controller) {
          var _a9;
          try {
            const finalUsage = usage != null ? usage : {
              promptTokens: NaN,
              completionTokens: NaN,
              totalTokens: NaN
            };
            const finalFinishReason = finishReason != null ? finishReason : "unknown";
            const telemetryToolCalls = toolCalls.length > 0 ? JSON.stringify(toolCalls) : void 0;
            doStreamSpan.setAttributes(
              selectTelemetryAttributes({
                telemetry,
                attributes: {
                  "ai.finishReason": finalFinishReason,
                  "ai.usage.promptTokens": finalUsage.promptTokens,
                  "ai.usage.completionTokens": finalUsage.completionTokens,
                  "ai.result.text": { output: () => text },
                  "ai.result.toolCalls": { output: () => telemetryToolCalls },
                  // standardized gen-ai llm span attributes:
                  "gen_ai.response.finish_reasons": [finalFinishReason],
                  "gen_ai.usage.prompt_tokens": finalUsage.promptTokens,
                  "gen_ai.usage.completion_tokens": finalUsage.completionTokens
                }
              })
            );
            doStreamSpan.end();
            rootSpan.setAttributes(
              selectTelemetryAttributes({
                telemetry,
                attributes: {
                  "ai.finishReason": finalFinishReason,
                  "ai.usage.promptTokens": finalUsage.promptTokens,
                  "ai.usage.completionTokens": finalUsage.completionTokens,
                  "ai.result.text": { output: () => text },
                  "ai.result.toolCalls": { output: () => telemetryToolCalls }
                }
              })
            );
            resolveToolResults(toolResults);
            await ((_a9 = self.onFinish) == null ? void 0 : _a9.call(self, {
              finishReason: finalFinishReason,
              usage: finalUsage,
              text,
              toolCalls,
              // The tool results are inferred as a never[] type, because they are
              // optional and the execute method with an inferred result type is
              // optional as well. Therefore we need to cast the toolResults to any.
              // The type exposed to the users will be correctly inferred.
              toolResults,
              rawResponse,
              warnings
            }));
          } catch (error) {
            controller.error(error);
          } finally {
            rootSpan.end();
          }
        }
      })
    );
  }
  /**
  Split out a new stream from the original stream.
  The original stream is replaced to allow for further splitting,
  since we do not know how many times the stream will be split.
  
  Note: this leads to buffering the stream content on the server.
  However, the LLM results are expected to be small enough to not cause issues.
     */
  teeStream() {
    const [stream1, stream2] = this.originalStream.tee();
    this.originalStream = stream2;
    return stream1;
  }
  get textStream() {
    return createAsyncIterableStream(this.teeStream(), {
      transform(chunk, controller) {
        if (chunk.type === "text-delta") {
          if (chunk.textDelta.length > 0) {
            controller.enqueue(chunk.textDelta);
          }
        } else if (chunk.type === "error") {
          controller.error(chunk.error);
        }
      }
    });
  }
  get fullStream() {
    return createAsyncIterableStream(this.teeStream(), {
      transform(chunk, controller) {
        if (chunk.type === "text-delta") {
          if (chunk.textDelta.length > 0) {
            controller.enqueue(chunk);
          }
        } else {
          controller.enqueue(chunk);
        }
      }
    });
  }
  toAIStream(callbacks = {}) {
    return this.toDataStream({ callbacks });
  }
  toDataStream({
    callbacks = {},
    getErrorMessage: getErrorMessage4 = () => ""
    // mask error messages for safety by default
  } = {}) {
    let aggregatedResponse = "";
    const callbackTransformer = new TransformStream({
      async start() {
        if (callbacks.onStart)
          await callbacks.onStart();
      },
      async transform(chunk, controller) {
        controller.enqueue(chunk);
        if (chunk.type === "text-delta") {
          const textDelta = chunk.textDelta;
          aggregatedResponse += textDelta;
          if (callbacks.onToken)
            await callbacks.onToken(textDelta);
          if (callbacks.onText)
            await callbacks.onText(textDelta);
        }
      },
      async flush() {
        if (callbacks.onCompletion)
          await callbacks.onCompletion(aggregatedResponse);
        if (callbacks.onFinal)
          await callbacks.onFinal(aggregatedResponse);
      }
    });
    const streamPartsTransformer = new TransformStream({
      transform: async (chunk, controller) => {
        const chunkType = chunk.type;
        switch (chunkType) {
          case "text-delta":
            controller.enqueue((0, import_ui_utils6.formatStreamPart)("text", chunk.textDelta));
            break;
          case "tool-call-streaming-start":
            controller.enqueue(
              (0, import_ui_utils6.formatStreamPart)("tool_call_streaming_start", {
                toolCallId: chunk.toolCallId,
                toolName: chunk.toolName
              })
            );
            break;
          case "tool-call-delta":
            controller.enqueue(
              (0, import_ui_utils6.formatStreamPart)("tool_call_delta", {
                toolCallId: chunk.toolCallId,
                argsTextDelta: chunk.argsTextDelta
              })
            );
            break;
          case "tool-call":
            controller.enqueue(
              (0, import_ui_utils6.formatStreamPart)("tool_call", {
                toolCallId: chunk.toolCallId,
                toolName: chunk.toolName,
                args: chunk.args
              })
            );
            break;
          case "tool-result":
            controller.enqueue(
              (0, import_ui_utils6.formatStreamPart)("tool_result", {
                toolCallId: chunk.toolCallId,
                result: chunk.result
              })
            );
            break;
          case "error":
            controller.enqueue(
              (0, import_ui_utils6.formatStreamPart)("error", getErrorMessage4(chunk.error))
            );
            break;
          case "finish":
            controller.enqueue(
              (0, import_ui_utils6.formatStreamPart)("finish_message", {
                finishReason: chunk.finishReason,
                usage: {
                  promptTokens: chunk.usage.promptTokens,
                  completionTokens: chunk.usage.completionTokens
                }
              })
            );
            break;
          default: {
            const exhaustiveCheck = chunkType;
            throw new Error(`Unknown chunk type: ${exhaustiveCheck}`);
          }
        }
      }
    });
    return this.fullStream.pipeThrough(callbackTransformer).pipeThrough(streamPartsTransformer).pipeThrough(new TextEncoderStream());
  }
  pipeAIStreamToResponse(response, init) {
    return this.pipeDataStreamToResponse(response, init);
  }
  pipeDataStreamToResponse(response, init) {
    var _a9;
    response.writeHead((_a9 = init == null ? void 0 : init.status) != null ? _a9 : 200, {
      "Content-Type": "text/plain; charset=utf-8",
      ...init == null ? void 0 : init.headers
    });
    const reader = this.toDataStream().getReader();
    const read = async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done)
            break;
          response.write(value);
        }
      } catch (error) {
        throw error;
      } finally {
        response.end();
      }
    };
    read();
  }
  pipeTextStreamToResponse(response, init) {
    var _a9;
    response.writeHead((_a9 = init == null ? void 0 : init.status) != null ? _a9 : 200, {
      "Content-Type": "text/plain; charset=utf-8",
      ...init == null ? void 0 : init.headers
    });
    const reader = this.textStream.pipeThrough(new TextEncoderStream()).getReader();
    const read = async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done)
            break;
          response.write(value);
        }
      } catch (error) {
        throw error;
      } finally {
        response.end();
      }
    };
    read();
  }
  toAIStreamResponse(options) {
    return this.toDataStreamResponse(options);
  }
  toDataStreamResponse(options) {
    var _a9;
    const init = options == null ? void 0 : "init" in options ? options.init : {
      headers: "headers" in options ? options.headers : void 0,
      status: "status" in options ? options.status : void 0,
      statusText: "statusText" in options ? options.statusText : void 0
    };
    const data = options == null ? void 0 : "data" in options ? options.data : void 0;
    const getErrorMessage4 = options == null ? void 0 : "getErrorMessage" in options ? options.getErrorMessage : void 0;
    const stream = data ? mergeStreams(data.stream, this.toDataStream({ getErrorMessage: getErrorMessage4 })) : this.toDataStream({ getErrorMessage: getErrorMessage4 });
    return new Response(stream, {
      status: (_a9 = init == null ? void 0 : init.status) != null ? _a9 : 200,
      statusText: init == null ? void 0 : init.statusText,
      headers: prepareResponseHeaders(init, {
        contentType: "text/plain; charset=utf-8",
        dataStreamVersion: "v1"
      })
    });
  }
  toTextStreamResponse(init) {
    var _a9;
    return new Response(this.textStream.pipeThrough(new TextEncoderStream()), {
      status: (_a9 = init == null ? void 0 : init.status) != null ? _a9 : 200,
      headers: prepareResponseHeaders(init, {
        contentType: "text/plain; charset=utf-8"
      })
    });
  }
};
var experimental_streamText = streamText;

// core/prompt/attachments-to-parts.ts
function attachmentsToParts(attachments) {
  var _a9, _b, _c;
  const parts = [];
  for (const attachment of attachments) {
    let url;
    try {
      url = new URL(attachment.url);
    } catch (error) {
      throw new Error(`Invalid URL: ${attachment.url}`);
    }
    switch (url.protocol) {
      case "http:":
      case "https:": {
        if ((_a9 = attachment.contentType) == null ? void 0 : _a9.startsWith("image/")) {
          parts.push({ type: "image", image: url });
        }
        break;
      }
      case "data:": {
        let header;
        let base64Content;
        let mimeType;
        try {
          [header, base64Content] = attachment.url.split(",");
          mimeType = header.split(";")[0].split(":")[1];
        } catch (error) {
          throw new Error(`Error processing data URL: ${attachment.url}`);
        }
        if (mimeType == null || base64Content == null) {
          throw new Error(`Invalid data URL format: ${attachment.url}`);
        }
        if ((_b = attachment.contentType) == null ? void 0 : _b.startsWith("image/")) {
          parts.push({
            type: "image",
            image: convertDataContentToUint8Array(base64Content)
          });
        } else if ((_c = attachment.contentType) == null ? void 0 : _c.startsWith("text/")) {
          parts.push({
            type: "text",
            text: convertUint8ArrayToText(
              convertDataContentToUint8Array(base64Content)
            )
          });
        }
        break;
      }
      default: {
        throw new Error(`Unsupported URL protocol: ${url.protocol}`);
      }
    }
  }
  return parts;
}

// core/prompt/convert-to-core-messages.ts
function convertToCoreMessages(messages) {
  const coreMessages = [];
  for (const {
    role,
    content,
    toolInvocations,
    experimental_attachments
  } of messages) {
    switch (role) {
      case "system": {
        coreMessages.push({
          role: "system",
          content
        });
        break;
      }
      case "user": {
        coreMessages.push({
          role: "user",
          content: experimental_attachments ? [
            { type: "text", text: content },
            ...attachmentsToParts(experimental_attachments)
          ] : content
        });
        break;
      }
      case "assistant": {
        if (toolInvocations == null) {
          coreMessages.push({ role: "assistant", content });
          break;
        }
        coreMessages.push({
          role: "assistant",
          content: [
            { type: "text", text: content },
            ...toolInvocations.map(({ toolCallId, toolName, args }) => ({
              type: "tool-call",
              toolCallId,
              toolName,
              args
            }))
          ]
        });
        coreMessages.push({
          role: "tool",
          content: toolInvocations.map(
            ({ toolCallId, toolName, args, result }) => ({
              type: "tool-result",
              toolCallId,
              toolName,
              args,
              result
            })
          )
        });
        break;
      }
      default: {
        const _exhaustiveCheck = role;
        throw new Error(`Unhandled role: ${_exhaustiveCheck}`);
      }
    }
  }
  return coreMessages;
}

// core/registry/invalid-model-id-error.ts
var InvalidModelIdError = class extends Error {
  constructor({
    id,
    message = `Invalid model id: ${id}`
  }) {
    super(message);
    this.name = "AI_InvalidModelIdError";
    this.id = id;
  }
  static isInvalidModelIdError(error) {
    return error instanceof Error && error.name === "AI_InvalidModelIdError" && typeof error.id === "string";
  }
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      stack: this.stack,
      id: this.id
    };
  }
};

// core/registry/no-such-model-error.ts
var NoSuchModelError = class extends Error {
  constructor({
    modelId,
    modelType,
    message = `No such ${modelType}: ${modelId}`
  }) {
    super(message);
    this.name = "AI_NoSuchModelError";
    this.modelId = modelId;
    this.modelType = modelType;
  }
  static isNoSuchModelError(error) {
    return error instanceof Error && error.name === "AI_NoSuchModelError" && typeof error.modelId === "string" && typeof error.modelType === "string";
  }
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      stack: this.stack,
      modelId: this.modelId,
      modelType: this.modelType
    };
  }
};

// core/registry/no-such-provider-error.ts
var NoSuchProviderError = class extends Error {
  constructor({
    providerId,
    availableProviders,
    message = `No such provider: ${providerId} (available providers: ${availableProviders.join()})`
  }) {
    super(message);
    this.name = "AI_NoSuchProviderError";
    this.providerId = providerId;
    this.availableProviders = availableProviders;
  }
  static isNoSuchProviderError(error) {
    return error instanceof Error && error.name === "AI_NoSuchProviderError" && typeof error.providerId === "string" && Array.isArray(error.availableProviders);
  }
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      stack: this.stack,
      providerId: this.providerId,
      availableProviders: this.availableProviders
    };
  }
};

// core/registry/provider-registry.ts
function experimental_createProviderRegistry(providers) {
  const registry = new DefaultProviderRegistry();
  for (const [id, provider] of Object.entries(providers)) {
    registry.registerProvider({ id, provider });
  }
  return registry;
}
var experimental_createModelRegistry = experimental_createProviderRegistry;
var DefaultProviderRegistry = class {
  constructor() {
    this.providers = {};
  }
  registerProvider({ id, provider }) {
    this.providers[id] = provider;
  }
  getProvider(id) {
    const provider = this.providers[id];
    if (provider == null) {
      throw new NoSuchProviderError({
        providerId: id,
        availableProviders: Object.keys(this.providers)
      });
    }
    return provider;
  }
  splitId(id) {
    const index = id.indexOf(":");
    if (index === -1) {
      throw new InvalidModelIdError({ id });
    }
    return [id.slice(0, index), id.slice(index + 1)];
  }
  languageModel(id) {
    var _a9, _b;
    const [providerId, modelId] = this.splitId(id);
    const model = (_b = (_a9 = this.getProvider(providerId)).languageModel) == null ? void 0 : _b.call(_a9, modelId);
    if (model == null) {
      throw new NoSuchModelError({ modelId: id, modelType: "language model" });
    }
    return model;
  }
  textEmbeddingModel(id) {
    var _a9, _b;
    const [providerId, modelId] = this.splitId(id);
    const model = (_b = (_a9 = this.getProvider(providerId)).textEmbedding) == null ? void 0 : _b.call(_a9, modelId);
    if (model == null) {
      throw new NoSuchModelError({
        modelId: id,
        modelType: "text embedding model"
      });
    }
    return model;
  }
};

// core/tool/tool.ts
function tool(tool2) {
  return tool2;
}

// core/util/cosine-similarity.ts
function cosineSimilarity(vector1, vector2) {
  if (vector1.length !== vector2.length) {
    throw new Error(
      `Vectors must have the same length (vector1: ${vector1.length} elements, vector2: ${vector2.length} elements)`
    );
  }
  return dotProduct(vector1, vector2) / (magnitude(vector1) * magnitude(vector2));
}
function dotProduct(vector1, vector2) {
  return vector1.reduce(
    (accumulator, value, index) => accumulator + value * vector2[index],
    0
  );
}
function magnitude(vector) {
  return Math.sqrt(dotProduct(vector, vector));
}

// errors/index.ts
var import_provider11 = require("@ai-sdk/provider");

// streams/ai-stream.ts
var import_eventsource_parser = require("eventsource-parser");
function createEventStreamTransformer(customParser) {
  const textDecoder = new TextDecoder();
  let eventSourceParser;
  return new TransformStream({
    async start(controller) {
      eventSourceParser = (0, import_eventsource_parser.createParser)(
        (event) => {
          if ("data" in event && event.type === "event" && event.data === "[DONE]" || // Replicate doesn't send [DONE] but does send a 'done' event
          // @see https://replicate.com/docs/streaming
          event.event === "done") {
            controller.terminate();
            return;
          }
          if ("data" in event) {
            const parsedMessage = customParser ? customParser(event.data, {
              event: event.event
            }) : event.data;
            if (parsedMessage)
              controller.enqueue(parsedMessage);
          }
        }
      );
    },
    transform(chunk) {
      eventSourceParser.feed(textDecoder.decode(chunk));
    }
  });
}
function createCallbacksTransformer(cb) {
  const textEncoder = new TextEncoder();
  let aggregatedResponse = "";
  const callbacks = cb || {};
  return new TransformStream({
    async start() {
      if (callbacks.onStart)
        await callbacks.onStart();
    },
    async transform(message, controller) {
      const content = typeof message === "string" ? message : message.content;
      controller.enqueue(textEncoder.encode(content));
      aggregatedResponse += content;
      if (callbacks.onToken)
        await callbacks.onToken(content);
      if (callbacks.onText && typeof message === "string") {
        await callbacks.onText(message);
      }
    },
    async flush() {
      const isOpenAICallbacks = isOfTypeOpenAIStreamCallbacks(callbacks);
      if (callbacks.onCompletion) {
        await callbacks.onCompletion(aggregatedResponse);
      }
      if (callbacks.onFinal && !isOpenAICallbacks) {
        await callbacks.onFinal(aggregatedResponse);
      }
    }
  });
}
function isOfTypeOpenAIStreamCallbacks(callbacks) {
  return "experimental_onFunctionCall" in callbacks;
}
function trimStartOfStreamHelper() {
  let isStreamStart = true;
  return (text) => {
    if (isStreamStart) {
      text = text.trimStart();
      if (text)
        isStreamStart = false;
    }
    return text;
  };
}
function AIStream(response, customParser, callbacks) {
  if (!response.ok) {
    if (response.body) {
      const reader = response.body.getReader();
      return new ReadableStream({
        async start(controller) {
          const { done, value } = await reader.read();
          if (!done) {
            const errorText = new TextDecoder().decode(value);
            controller.error(new Error(`Response error: ${errorText}`));
          }
        }
      });
    } else {
      return new ReadableStream({
        start(controller) {
          controller.error(new Error("Response error: No response body"));
        }
      });
    }
  }
  const responseBodyStream = response.body || createEmptyReadableStream();
  return responseBodyStream.pipeThrough(createEventStreamTransformer(customParser)).pipeThrough(createCallbacksTransformer(callbacks));
}
function createEmptyReadableStream() {
  return new ReadableStream({
    start(controller) {
      controller.close();
    }
  });
}
function readableFromAsyncIterable(iterable) {
  let it = iterable[Symbol.asyncIterator]();
  return new ReadableStream({
    async pull(controller) {
      const { done, value } = await it.next();
      if (done)
        controller.close();
      else
        controller.enqueue(value);
    },
    async cancel(reason) {
      var _a9;
      await ((_a9 = it.return) == null ? void 0 : _a9.call(it, reason));
    }
  });
}

// streams/stream-data.ts
var import_ui_utils3 = require("@ai-sdk/ui-utils");
var STREAM_DATA_WARNING_TIME_MS = 15 * 1e3;
var StreamData2 = class {
  constructor() {
    this.encoder = new TextEncoder();
    this.controller = null;
    this.isClosed = false;
    this.warningTimeout = null;
    const self = this;
    this.stream = new ReadableStream({
      start: async (controller) => {
        self.controller = controller;
        if (process.env.NODE_ENV === "development") {
          self.warningTimeout = setTimeout(() => {
            console.warn(
              "The data stream is hanging. Did you forget to close it with `data.close()`?"
            );
          }, STREAM_DATA_WARNING_TIME_MS);
        }
      },
      pull: (controller) => {
      },
      cancel: (reason) => {
        this.isClosed = true;
      }
    });
  }
  async close() {
    if (this.isClosed) {
      throw new Error("Data Stream has already been closed.");
    }
    if (!this.controller) {
      throw new Error("Stream controller is not initialized.");
    }
    this.controller.close();
    this.isClosed = true;
    if (this.warningTimeout) {
      clearTimeout(this.warningTimeout);
    }
  }
  append(value) {
    if (this.isClosed) {
      throw new Error("Data Stream has already been closed.");
    }
    if (!this.controller) {
      throw new Error("Stream controller is not initialized.");
    }
    this.controller.enqueue(
      this.encoder.encode((0, import_ui_utils3.formatStreamPart)("data", [value]))
    );
  }
  appendMessageAnnotation(value) {
    if (this.isClosed) {
      throw new Error("Data Stream has already been closed.");
    }
    if (!this.controller) {
      throw new Error("Stream controller is not initialized.");
    }
    this.controller.enqueue(
      this.encoder.encode((0, import_ui_utils3.formatStreamPart)("message_annotations", [value]))
    );
  }
};
function createStreamDataTransformer() {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  return new TransformStream({
    transform: async (chunk, controller) => {
      const message = decoder.decode(chunk);
      controller.enqueue(encoder.encode((0, import_ui_utils3.formatStreamPart)("text", message)));
    }
  });
}
var experimental_StreamData = class extends StreamData2 {
};

// streams/anthropic-stream.ts
function parseAnthropicStream() {
  let previous = "";
  return (data) => {
    const json = JSON.parse(data);
    if ("error" in json) {
      throw new Error(`${json.error.type}: ${json.error.message}`);
    }
    if (!("completion" in json)) {
      return;
    }
    const text = json.completion;
    if (!previous || text.length > previous.length && text.startsWith(previous)) {
      const delta = text.slice(previous.length);
      previous = text;
      return delta;
    }
    return text;
  };
}
async function* streamable(stream) {
  for await (const chunk of stream) {
    if ("completion" in chunk) {
      const text = chunk.completion;
      if (text)
        yield text;
    } else if ("delta" in chunk) {
      const { delta } = chunk;
      if ("text" in delta) {
        const text = delta.text;
        if (text)
          yield text;
      }
    }
  }
}
function AnthropicStream(res, cb) {
  if (Symbol.asyncIterator in res) {
    return readableFromAsyncIterable(streamable(res)).pipeThrough(createCallbacksTransformer(cb)).pipeThrough(createStreamDataTransformer());
  } else {
    return AIStream(res, parseAnthropicStream(), cb).pipeThrough(
      createStreamDataTransformer()
    );
  }
}

// streams/assistant-response.ts
var import_ui_utils4 = require("@ai-sdk/ui-utils");
function AssistantResponse({ threadId, messageId }, process2) {
  const stream = new ReadableStream({
    async start(controller) {
      var _a9;
      const textEncoder = new TextEncoder();
      const sendMessage = (message) => {
        controller.enqueue(
          textEncoder.encode((0, import_ui_utils4.formatStreamPart)("assistant_message", message))
        );
      };
      const sendDataMessage = (message) => {
        controller.enqueue(
          textEncoder.encode((0, import_ui_utils4.formatStreamPart)("data_message", message))
        );
      };
      const sendError = (errorMessage) => {
        controller.enqueue(
          textEncoder.encode((0, import_ui_utils4.formatStreamPart)("error", errorMessage))
        );
      };
      const forwardStream = async (stream2) => {
        var _a10, _b;
        let result = void 0;
        for await (const value of stream2) {
          switch (value.event) {
            case "thread.message.created": {
              controller.enqueue(
                textEncoder.encode(
                  (0, import_ui_utils4.formatStreamPart)("assistant_message", {
                    id: value.data.id,
                    role: "assistant",
                    content: [{ type: "text", text: { value: "" } }]
                  })
                )
              );
              break;
            }
            case "thread.message.delta": {
              const content = (_a10 = value.data.delta.content) == null ? void 0 : _a10[0];
              if ((content == null ? void 0 : content.type) === "text" && ((_b = content.text) == null ? void 0 : _b.value) != null) {
                controller.enqueue(
                  textEncoder.encode(
                    (0, import_ui_utils4.formatStreamPart)("text", content.text.value)
                  )
                );
              }
              break;
            }
            case "thread.run.completed":
            case "thread.run.requires_action": {
              result = value.data;
              break;
            }
          }
        }
        return result;
      };
      controller.enqueue(
        textEncoder.encode(
          (0, import_ui_utils4.formatStreamPart)("assistant_control_data", {
            threadId,
            messageId
          })
        )
      );
      try {
        await process2({
          threadId,
          messageId,
          sendMessage,
          sendDataMessage,
          forwardStream
        });
      } catch (error) {
        sendError((_a9 = error.message) != null ? _a9 : `${error}`);
      } finally {
        controller.close();
      }
    },
    pull(controller) {
    },
    cancel() {
    }
  });
  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/plain; charset=utf-8"
    }
  });
}
var experimental_AssistantResponse = AssistantResponse;

// streams/aws-bedrock-stream.ts
async function* asDeltaIterable(response, extractTextDeltaFromChunk) {
  var _a9, _b;
  const decoder = new TextDecoder();
  for await (const chunk of (_a9 = response.body) != null ? _a9 : []) {
    const bytes = (_b = chunk.chunk) == null ? void 0 : _b.bytes;
    if (bytes != null) {
      const chunkText = decoder.decode(bytes);
      const chunkJSON = JSON.parse(chunkText);
      const delta = extractTextDeltaFromChunk(chunkJSON);
      if (delta != null) {
        yield delta;
      }
    }
  }
}
function AWSBedrockAnthropicMessagesStream(response, callbacks) {
  return AWSBedrockStream(response, callbacks, (chunk) => {
    var _a9;
    return (_a9 = chunk.delta) == null ? void 0 : _a9.text;
  });
}
function AWSBedrockAnthropicStream(response, callbacks) {
  return AWSBedrockStream(response, callbacks, (chunk) => chunk.completion);
}
function AWSBedrockCohereStream(response, callbacks) {
  return AWSBedrockStream(response, callbacks, (chunk) => chunk == null ? void 0 : chunk.text);
}
function AWSBedrockLlama2Stream(response, callbacks) {
  return AWSBedrockStream(response, callbacks, (chunk) => chunk.generation);
}
function AWSBedrockStream(response, callbacks, extractTextDeltaFromChunk) {
  return readableFromAsyncIterable(
    asDeltaIterable(response, extractTextDeltaFromChunk)
  ).pipeThrough(createCallbacksTransformer(callbacks)).pipeThrough(createStreamDataTransformer());
}

// streams/cohere-stream.ts
var utf8Decoder = new TextDecoder("utf-8");
async function processLines(lines, controller) {
  for (const line of lines) {
    const { text, is_finished } = JSON.parse(line);
    if (!is_finished) {
      controller.enqueue(text);
    }
  }
}
async function readAndProcessLines(reader, controller) {
  let segment = "";
  while (true) {
    const { value: chunk, done } = await reader.read();
    if (done) {
      break;
    }
    segment += utf8Decoder.decode(chunk, { stream: true });
    const linesArray = segment.split(/\r\n|\n|\r/g);
    segment = linesArray.pop() || "";
    await processLines(linesArray, controller);
  }
  if (segment) {
    const linesArray = [segment];
    await processLines(linesArray, controller);
  }
  controller.close();
}
function createParser2(res) {
  var _a9;
  const reader = (_a9 = res.body) == null ? void 0 : _a9.getReader();
  return new ReadableStream({
    async start(controller) {
      if (!reader) {
        controller.close();
        return;
      }
      await readAndProcessLines(reader, controller);
    }
  });
}
async function* streamable2(stream) {
  for await (const chunk of stream) {
    if (chunk.eventType === "text-generation") {
      const text = chunk.text;
      if (text)
        yield text;
    }
  }
}
function CohereStream(reader, callbacks) {
  if (Symbol.asyncIterator in reader) {
    return readableFromAsyncIterable(streamable2(reader)).pipeThrough(createCallbacksTransformer(callbacks)).pipeThrough(createStreamDataTransformer());
  } else {
    return createParser2(reader).pipeThrough(createCallbacksTransformer(callbacks)).pipeThrough(createStreamDataTransformer());
  }
}

// streams/google-generative-ai-stream.ts
async function* streamable3(response) {
  var _a9, _b, _c;
  for await (const chunk of response.stream) {
    const parts = (_c = (_b = (_a9 = chunk.candidates) == null ? void 0 : _a9[0]) == null ? void 0 : _b.content) == null ? void 0 : _c.parts;
    if (parts === void 0) {
      continue;
    }
    const firstPart = parts[0];
    if (typeof firstPart.text === "string") {
      yield firstPart.text;
    }
  }
}
function GoogleGenerativeAIStream(response, cb) {
  return readableFromAsyncIterable(streamable3(response)).pipeThrough(createCallbacksTransformer(cb)).pipeThrough(createStreamDataTransformer());
}

// streams/huggingface-stream.ts
function createParser3(res) {
  const trimStartOfStream = trimStartOfStreamHelper();
  return new ReadableStream({
    async pull(controller) {
      var _a9, _b;
      const { value, done } = await res.next();
      if (done) {
        controller.close();
        return;
      }
      const text = trimStartOfStream((_b = (_a9 = value.token) == null ? void 0 : _a9.text) != null ? _b : "");
      if (!text)
        return;
      if (value.generated_text != null && value.generated_text.length > 0) {
        return;
      }
      if (text === "</s>" || text === "<|endoftext|>" || text === "<|end|>") {
        return;
      }
      controller.enqueue(text);
    }
  });
}
function HuggingFaceStream(res, callbacks) {
  return createParser3(res).pipeThrough(createCallbacksTransformer(callbacks)).pipeThrough(createStreamDataTransformer());
}

// streams/inkeep-stream.ts
function InkeepStream(res, callbacks) {
  if (!res.body) {
    throw new Error("Response body is null");
  }
  let chat_session_id = "";
  let records_cited;
  const inkeepEventParser = (data, options) => {
    var _a9, _b;
    const { event } = options;
    if (event === "records_cited") {
      records_cited = JSON.parse(data);
      (_a9 = callbacks == null ? void 0 : callbacks.onRecordsCited) == null ? void 0 : _a9.call(callbacks, records_cited);
    }
    if (event === "message_chunk") {
      const inkeepMessageChunk = JSON.parse(data);
      chat_session_id = (_b = inkeepMessageChunk.chat_session_id) != null ? _b : chat_session_id;
      return inkeepMessageChunk.content_chunk;
    }
    return;
  };
  let { onRecordsCited, ...passThroughCallbacks } = callbacks || {};
  passThroughCallbacks = {
    ...passThroughCallbacks,
    onFinal: (completion) => {
      var _a9;
      const inkeepOnFinalMetadata = {
        chat_session_id,
        records_cited
      };
      (_a9 = callbacks == null ? void 0 : callbacks.onFinal) == null ? void 0 : _a9.call(callbacks, completion, inkeepOnFinalMetadata);
    }
  };
  return AIStream(res, inkeepEventParser, passThroughCallbacks).pipeThrough(
    createStreamDataTransformer()
  );
}

// streams/langchain-adapter.ts
var langchain_adapter_exports = {};
__export(langchain_adapter_exports, {
  toAIStream: () => toAIStream,
  toDataStream: () => toDataStream,
  toDataStreamResponse: () => toDataStreamResponse
});
function toAIStream(stream, callbacks) {
  return toDataStream(stream, callbacks);
}
function toDataStream(stream, callbacks) {
  return stream.pipeThrough(
    new TransformStream({
      transform: async (value, controller) => {
        var _a9;
        if (typeof value === "string") {
          controller.enqueue(value);
          return;
        }
        if ("event" in value) {
          if (value.event === "on_chat_model_stream") {
            forwardAIMessageChunk(
              (_a9 = value.data) == null ? void 0 : _a9.chunk,
              controller
            );
          }
          return;
        }
        forwardAIMessageChunk(value, controller);
      }
    })
  ).pipeThrough(createCallbacksTransformer(callbacks)).pipeThrough(createStreamDataTransformer());
}
function toDataStreamResponse(stream, options) {
  var _a9;
  const dataStream = toDataStream(stream, options == null ? void 0 : options.callbacks);
  const data = options == null ? void 0 : options.data;
  const init = options == null ? void 0 : options.init;
  const responseStream = data ? mergeStreams(data.stream, dataStream) : dataStream;
  return new Response(responseStream, {
    status: (_a9 = init == null ? void 0 : init.status) != null ? _a9 : 200,
    statusText: init == null ? void 0 : init.statusText,
    headers: prepareResponseHeaders(init, {
      contentType: "text/plain; charset=utf-8",
      dataStreamVersion: "v1"
    })
  });
}
function forwardAIMessageChunk(chunk, controller) {
  if (typeof chunk.content === "string") {
    controller.enqueue(chunk.content);
  } else {
    const content = chunk.content;
    for (const item of content) {
      if (item.type === "text") {
        controller.enqueue(item.text);
      }
    }
  }
}

// streams/langchain-stream.ts
function LangChainStream(callbacks) {
  const stream = new TransformStream();
  const writer = stream.writable.getWriter();
  const runs = /* @__PURE__ */ new Set();
  const handleError = async (e, runId) => {
    runs.delete(runId);
    await writer.ready;
    await writer.abort(e);
  };
  const handleStart = async (runId) => {
    runs.add(runId);
  };
  const handleEnd = async (runId) => {
    runs.delete(runId);
    if (runs.size === 0) {
      await writer.ready;
      await writer.close();
    }
  };
  return {
    stream: stream.readable.pipeThrough(createCallbacksTransformer(callbacks)).pipeThrough(createStreamDataTransformer()),
    writer,
    handlers: {
      handleLLMNewToken: async (token) => {
        await writer.ready;
        await writer.write(token);
      },
      handleLLMStart: async (_llm, _prompts, runId) => {
        handleStart(runId);
      },
      handleLLMEnd: async (_output, runId) => {
        await handleEnd(runId);
      },
      handleLLMError: async (e, runId) => {
        await handleError(e, runId);
      },
      handleChainStart: async (_chain, _inputs, runId) => {
        handleStart(runId);
      },
      handleChainEnd: async (_outputs, runId) => {
        await handleEnd(runId);
      },
      handleChainError: async (e, runId) => {
        await handleError(e, runId);
      },
      handleToolStart: async (_tool, _input, runId) => {
        handleStart(runId);
      },
      handleToolEnd: async (_output, runId) => {
        await handleEnd(runId);
      },
      handleToolError: async (e, runId) => {
        await handleError(e, runId);
      }
    }
  };
}

// streams/mistral-stream.ts
async function* streamable4(stream) {
  var _a9, _b;
  for await (const chunk of stream) {
    const content = (_b = (_a9 = chunk.choices[0]) == null ? void 0 : _a9.delta) == null ? void 0 : _b.content;
    if (content === void 0 || content === "") {
      continue;
    }
    yield content;
  }
}
function MistralStream(response, callbacks) {
  const stream = readableFromAsyncIterable(streamable4(response));
  return stream.pipeThrough(createCallbacksTransformer(callbacks)).pipeThrough(createStreamDataTransformer());
}

// streams/openai-stream.ts
var import_ui_utils5 = require("@ai-sdk/ui-utils");
function parseOpenAIStream() {
  const extract = chunkToText();
  return (data) => extract(JSON.parse(data));
}
async function* streamable5(stream) {
  const extract = chunkToText();
  for await (let chunk of stream) {
    if ("promptFilterResults" in chunk) {
      chunk = {
        id: chunk.id,
        created: chunk.created.getDate(),
        object: chunk.object,
        // not exposed by Azure API
        model: chunk.model,
        // not exposed by Azure API
        choices: chunk.choices.map((choice) => {
          var _a9, _b, _c, _d, _e, _f, _g;
          return {
            delta: {
              content: (_a9 = choice.delta) == null ? void 0 : _a9.content,
              function_call: (_b = choice.delta) == null ? void 0 : _b.functionCall,
              role: (_c = choice.delta) == null ? void 0 : _c.role,
              tool_calls: ((_e = (_d = choice.delta) == null ? void 0 : _d.toolCalls) == null ? void 0 : _e.length) ? (_g = (_f = choice.delta) == null ? void 0 : _f.toolCalls) == null ? void 0 : _g.map((toolCall, index) => ({
                index,
                id: toolCall.id,
                function: toolCall.function,
                type: toolCall.type
              })) : void 0
            },
            finish_reason: choice.finishReason,
            index: choice.index
          };
        })
      };
    }
    const text = extract(chunk);
    if (text)
      yield text;
  }
}
function chunkToText() {
  const trimStartOfStream = trimStartOfStreamHelper();
  let isFunctionStreamingIn;
  return (json) => {
    var _a9, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, _l, _m, _n, _o, _p, _q, _r;
    if (isChatCompletionChunk(json)) {
      const delta = (_a9 = json.choices[0]) == null ? void 0 : _a9.delta;
      if ((_b = delta.function_call) == null ? void 0 : _b.name) {
        isFunctionStreamingIn = true;
        return {
          isText: false,
          content: `{"function_call": {"name": "${delta.function_call.name}", "arguments": "`
        };
      } else if ((_e = (_d = (_c = delta.tool_calls) == null ? void 0 : _c[0]) == null ? void 0 : _d.function) == null ? void 0 : _e.name) {
        isFunctionStreamingIn = true;
        const toolCall = delta.tool_calls[0];
        if (toolCall.index === 0) {
          return {
            isText: false,
            content: `{"tool_calls":[ {"id": "${toolCall.id}", "type": "function", "function": {"name": "${(_f = toolCall.function) == null ? void 0 : _f.name}", "arguments": "`
          };
        } else {
          return {
            isText: false,
            content: `"}}, {"id": "${toolCall.id}", "type": "function", "function": {"name": "${(_g = toolCall.function) == null ? void 0 : _g.name}", "arguments": "`
          };
        }
      } else if ((_h = delta.function_call) == null ? void 0 : _h.arguments) {
        return {
          isText: false,
          content: cleanupArguments((_i = delta.function_call) == null ? void 0 : _i.arguments)
        };
      } else if ((_l = (_k = (_j = delta.tool_calls) == null ? void 0 : _j[0]) == null ? void 0 : _k.function) == null ? void 0 : _l.arguments) {
        return {
          isText: false,
          content: cleanupArguments((_o = (_n = (_m = delta.tool_calls) == null ? void 0 : _m[0]) == null ? void 0 : _n.function) == null ? void 0 : _o.arguments)
        };
      } else if (isFunctionStreamingIn && (((_p = json.choices[0]) == null ? void 0 : _p.finish_reason) === "function_call" || ((_q = json.choices[0]) == null ? void 0 : _q.finish_reason) === "stop")) {
        isFunctionStreamingIn = false;
        return {
          isText: false,
          content: '"}}'
        };
      } else if (isFunctionStreamingIn && ((_r = json.choices[0]) == null ? void 0 : _r.finish_reason) === "tool_calls") {
        isFunctionStreamingIn = false;
        return {
          isText: false,
          content: '"}}]}'
        };
      }
    }
    const text = trimStartOfStream(
      isChatCompletionChunk(json) && json.choices[0].delta.content ? json.choices[0].delta.content : isCompletion(json) ? json.choices[0].text : ""
    );
    return text;
  };
  function cleanupArguments(argumentChunk) {
    let escapedPartialJson = argumentChunk.replace(/\\/g, "\\\\").replace(/\//g, "\\/").replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t").replace(/\f/g, "\\f");
    return `${escapedPartialJson}`;
  }
}
var __internal__OpenAIFnMessagesSymbol = Symbol(
  "internal_openai_fn_messages"
);
function isChatCompletionChunk(data) {
  return "choices" in data && data.choices && data.choices[0] && "delta" in data.choices[0];
}
function isCompletion(data) {
  return "choices" in data && data.choices && data.choices[0] && "text" in data.choices[0];
}
function OpenAIStream(res, callbacks) {
  const cb = callbacks;
  let stream;
  if (Symbol.asyncIterator in res) {
    stream = readableFromAsyncIterable(streamable5(res)).pipeThrough(
      createCallbacksTransformer(
        (cb == null ? void 0 : cb.experimental_onFunctionCall) || (cb == null ? void 0 : cb.experimental_onToolCall) ? {
          ...cb,
          onFinal: void 0
        } : {
          ...cb
        }
      )
    );
  } else {
    stream = AIStream(
      res,
      parseOpenAIStream(),
      (cb == null ? void 0 : cb.experimental_onFunctionCall) || (cb == null ? void 0 : cb.experimental_onToolCall) ? {
        ...cb,
        onFinal: void 0
      } : {
        ...cb
      }
    );
  }
  if (cb && (cb.experimental_onFunctionCall || cb.experimental_onToolCall)) {
    const functionCallTransformer = createFunctionCallTransformer(cb);
    return stream.pipeThrough(functionCallTransformer);
  } else {
    return stream.pipeThrough(createStreamDataTransformer());
  }
}
function createFunctionCallTransformer(callbacks) {
  const textEncoder = new TextEncoder();
  let isFirstChunk = true;
  let aggregatedResponse = "";
  let aggregatedFinalCompletionResponse = "";
  let isFunctionStreamingIn = false;
  let functionCallMessages = callbacks[__internal__OpenAIFnMessagesSymbol] || [];
  const decode = (0, import_ui_utils5.createChunkDecoder)();
  return new TransformStream({
    async transform(chunk, controller) {
      const message = decode(chunk);
      aggregatedFinalCompletionResponse += message;
      const shouldHandleAsFunction = isFirstChunk && (message.startsWith('{"function_call":') || message.startsWith('{"tool_calls":'));
      if (shouldHandleAsFunction) {
        isFunctionStreamingIn = true;
        aggregatedResponse += message;
        isFirstChunk = false;
        return;
      }
      if (!isFunctionStreamingIn) {
        controller.enqueue(
          textEncoder.encode((0, import_ui_utils5.formatStreamPart)("text", message))
        );
        return;
      } else {
        aggregatedResponse += message;
      }
    },
    async flush(controller) {
      try {
        if (!isFirstChunk && isFunctionStreamingIn && (callbacks.experimental_onFunctionCall || callbacks.experimental_onToolCall)) {
          isFunctionStreamingIn = false;
          const payload = JSON.parse(aggregatedResponse);
          let newFunctionCallMessages = [
            ...functionCallMessages
          ];
          let functionResponse = void 0;
          if (callbacks.experimental_onFunctionCall) {
            if (payload.function_call === void 0) {
              console.warn(
                "experimental_onFunctionCall should not be defined when using tools"
              );
            }
            const argumentsPayload = JSON.parse(
              payload.function_call.arguments
            );
            functionResponse = await callbacks.experimental_onFunctionCall(
              {
                name: payload.function_call.name,
                arguments: argumentsPayload
              },
              (result) => {
                newFunctionCallMessages = [
                  ...functionCallMessages,
                  {
                    role: "assistant",
                    content: "",
                    function_call: payload.function_call
                  },
                  {
                    role: "function",
                    name: payload.function_call.name,
                    content: JSON.stringify(result)
                  }
                ];
                return newFunctionCallMessages;
              }
            );
          }
          if (callbacks.experimental_onToolCall) {
            const toolCalls = {
              tools: []
            };
            for (const tool2 of payload.tool_calls) {
              toolCalls.tools.push({
                id: tool2.id,
                type: "function",
                func: {
                  name: tool2.function.name,
                  arguments: JSON.parse(tool2.function.arguments)
                }
              });
            }
            let responseIndex = 0;
            try {
              functionResponse = await callbacks.experimental_onToolCall(
                toolCalls,
                (result) => {
                  if (result) {
                    const { tool_call_id, function_name, tool_call_result } = result;
                    newFunctionCallMessages = [
                      ...newFunctionCallMessages,
                      // Only append the assistant message if it's the first response
                      ...responseIndex === 0 ? [
                        {
                          role: "assistant",
                          content: "",
                          tool_calls: payload.tool_calls.map(
                            (tc) => ({
                              id: tc.id,
                              type: "function",
                              function: {
                                name: tc.function.name,
                                // we send the arguments an object to the user, but as the API expects a string, we need to stringify it
                                arguments: JSON.stringify(
                                  tc.function.arguments
                                )
                              }
                            })
                          )
                        }
                      ] : [],
                      // Append the function call result message
                      {
                        role: "tool",
                        tool_call_id,
                        name: function_name,
                        content: JSON.stringify(tool_call_result)
                      }
                    ];
                    responseIndex++;
                  }
                  return newFunctionCallMessages;
                }
              );
            } catch (e) {
              console.error("Error calling experimental_onToolCall:", e);
            }
          }
          if (!functionResponse) {
            controller.enqueue(
              textEncoder.encode(
                (0, import_ui_utils5.formatStreamPart)(
                  payload.function_call ? "function_call" : "tool_calls",
                  // parse to prevent double-encoding:
                  JSON.parse(aggregatedResponse)
                )
              )
            );
            return;
          } else if (typeof functionResponse === "string") {
            controller.enqueue(
              textEncoder.encode((0, import_ui_utils5.formatStreamPart)("text", functionResponse))
            );
            aggregatedFinalCompletionResponse = functionResponse;
            return;
          }
          const filteredCallbacks = {
            ...callbacks,
            onStart: void 0
          };
          callbacks.onFinal = void 0;
          const openAIStream = OpenAIStream(functionResponse, {
            ...filteredCallbacks,
            [__internal__OpenAIFnMessagesSymbol]: newFunctionCallMessages
          });
          const reader = openAIStream.getReader();
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }
            controller.enqueue(value);
          }
        }
      } finally {
        if (callbacks.onFinal && aggregatedFinalCompletionResponse) {
          await callbacks.onFinal(aggregatedFinalCompletionResponse);
        }
      }
    }
  });
}

// streams/replicate-stream.ts
async function ReplicateStream(res, cb, options) {
  var _a9;
  const url = (_a9 = res.urls) == null ? void 0 : _a9.stream;
  if (!url) {
    if (res.error)
      throw new Error(res.error);
    else
      throw new Error("Missing stream URL in Replicate response");
  }
  const eventStream = await fetch(url, {
    method: "GET",
    headers: {
      Accept: "text/event-stream",
      ...options == null ? void 0 : options.headers
    }
  });
  return AIStream(eventStream, void 0, cb).pipeThrough(
    createStreamDataTransformer()
  );
}

// streams/stream-to-response.ts
function streamToResponse(res, response, init, data) {
  var _a9;
  response.writeHead((_a9 = init == null ? void 0 : init.status) != null ? _a9 : 200, {
    "Content-Type": "text/plain; charset=utf-8",
    ...init == null ? void 0 : init.headers
  });
  let processedStream = res;
  if (data) {
    processedStream = mergeStreams(data.stream, res);
  }
  const reader = processedStream.getReader();
  function read() {
    reader.read().then(({ done, value }) => {
      if (done) {
        response.end();
        return;
      }
      response.write(value);
      read();
    });
  }
  read();
}

// streams/streaming-text-response.ts
var StreamingTextResponse = class extends Response {
  constructor(res, init, data) {
    let processedStream = res;
    if (data) {
      processedStream = mergeStreams(data.stream, res);
    }
    super(processedStream, {
      ...init,
      status: 200,
      headers: prepareResponseHeaders(init, {
        contentType: "text/plain; charset=utf-8"
      })
    });
  }
};

// streams/index.ts
var generateId2 = import_provider_utils8.generateId;
var nanoid = import_provider_utils8.generateId;
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  AISDKError,
  AIStream,
  APICallError,
  AWSBedrockAnthropicMessagesStream,
  AWSBedrockAnthropicStream,
  AWSBedrockCohereStream,
  AWSBedrockLlama2Stream,
  AWSBedrockStream,
  AnthropicStream,
  AssistantResponse,
  CohereStream,
  DownloadError,
  EmptyResponseBodyError,
  GoogleGenerativeAIStream,
  HuggingFaceStream,
  InkeepStream,
  InvalidArgumentError,
  InvalidDataContentError,
  InvalidMessageRoleError,
  InvalidModelIdError,
  InvalidPromptError,
  InvalidResponseDataError,
  InvalidToolArgumentsError,
  JSONParseError,
  LangChainAdapter,
  LangChainStream,
  LoadAPIKeyError,
  MistralStream,
  NoObjectGeneratedError,
  NoSuchModelError,
  NoSuchProviderError,
  NoSuchToolError,
  OpenAIStream,
  ReplicateStream,
  RetryError,
  StreamData,
  StreamingTextResponse,
  TypeValidationError,
  UnsupportedFunctionalityError,
  convertDataContentToBase64String,
  convertDataContentToUint8Array,
  convertToCoreMessages,
  convertUint8ArrayToText,
  cosineSimilarity,
  createCallbacksTransformer,
  createEventStreamTransformer,
  createStreamDataTransformer,
  embed,
  embedMany,
  experimental_AssistantResponse,
  experimental_StreamData,
  experimental_createModelRegistry,
  experimental_createProviderRegistry,
  experimental_generateObject,
  experimental_generateText,
  experimental_streamObject,
  experimental_streamText,
  formatStreamPart,
  generateId,
  generateObject,
  generateText,
  jsonSchema,
  nanoid,
  parseComplexResponse,
  parseStreamPart,
  readDataStream,
  readableFromAsyncIterable,
  streamObject,
  streamText,
  streamToResponse,
  tool,
  trimStartOfStreamHelper
});
//# sourceMappingURL=index.js.map