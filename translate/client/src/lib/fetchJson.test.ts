import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchJson } from './fetchJson';

function mockFetch(response: { ok: boolean; statusText?: string; json: () => Promise<unknown> }) {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('fetchJson', () => {
  it('returns the parsed JSON body on a successful response', async () => {
    mockFetch({ ok: true, json: () => Promise.resolve({ id: 1, name: 'job' }) });
    const result = await fetchJson<{ id: number; name: string }>('/api/whatever');
    expect(result).toEqual({ id: 1, name: 'job' });
  });

  it('throws the body\'s error message when the response is not ok', async () => {
    mockFetch({ ok: false, statusText: 'Bad Request', json: () => Promise.resolve({ error: 'invalid input' }) });
    await expect(fetchJson('/api/whatever')).rejects.toThrow('invalid input');
  });

  it('falls back to statusText when the error body has no error field', async () => {
    mockFetch({ ok: false, statusText: 'Internal Server Error', json: () => Promise.resolve({}) });
    await expect(fetchJson('/api/whatever')).rejects.toThrow('Internal Server Error');
  });

  it('falls back to statusText when the body is not valid JSON', async () => {
    mockFetch({ ok: false, statusText: 'Not Found', json: () => Promise.reject(new Error('not json')) });
    await expect(fetchJson('/api/whatever')).rejects.toThrow('Not Found');
  });

  it('forwards the init argument to fetch', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({}) });
    vi.stubGlobal('fetch', fetchMock);
    const init = { method: 'PATCH', body: '{}' };
    await fetchJson('/api/whatever', init);
    expect(fetchMock).toHaveBeenCalledWith('/api/whatever', init);
  });
});
