export type TranscriptionProvider = 'openai' | 'groq' | 'deepgram';

export interface TranscriptionOptions {
  provider: TranscriptionProvider;
  apiKey: string;
  model?: string;
  language?: string;
  prompt?: string;
  temperature?: number;
  fetchImpl?: typeof fetch;
}

export interface TranscriptionResult {
  provider: TranscriptionProvider;
  model: string;
  text: string;
  durationSeconds?: number;
}

export async function transcribeAudio(
  audioBlob: Blob,
  filename: string,
  options: TranscriptionOptions
): Promise<TranscriptionResult> {
  const fetchImpl = options.fetchImpl || globalThis.fetch;

  if (options.provider === 'deepgram') {
    const model = options.model || 'nova-2';
    const params = new URLSearchParams({ model, smart_format: 'true' });
    if (options.language) params.set('language', options.language);

    const res = await fetchImpl(`https://api.deepgram.com/v1/listen?${params.toString()}`, {
      method: 'POST',
      headers: {
        Authorization: `Token ${options.apiKey}`,
        'Content-Type': audioBlob.type || 'audio/wav',
      },
      body: audioBlob,
    });
    if (!res.ok) throw new Error(`Deepgram API Error (${res.status})`);
    const data = await res.json();
    const text = data?.results?.channels?.[0]?.alternatives?.[0]?.transcript || '';
    return { provider: 'deepgram', model, text, durationSeconds: data?.metadata?.duration };
  }

  const isGroq = options.provider === 'groq';
  const endpoint = isGroq
    ? 'https://api.groq.com/openai/v1/audio/transcriptions'
    : 'https://api.openai.com/v1/audio/transcriptions';
  const model = options.model || (isGroq ? 'whisper-large-v3-turbo' : 'whisper-1');

  const form = new FormData();
  form.append('file', audioBlob, filename);
  form.append('model', model);
  form.append('response_format', 'verbose_json');
  if (options.language) form.append('language', options.language);
  if (options.prompt) form.append('prompt', options.prompt);
  if (options.temperature !== undefined) form.append('temperature', String(options.temperature));

  const res = await fetchImpl(endpoint, {
    method: 'POST',
    headers: { Authorization: `Bearer ${options.apiKey}` },
    body: form,
  });
  if (!res.ok) throw new Error(`${options.provider} Transcription Error (${res.status})`);
  const data = await res.json();
  return {
    provider: options.provider,
    model,
    text: data.text || '',
    durationSeconds: data.duration,
  };
}
