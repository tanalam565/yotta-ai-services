jest.mock('axios', () => ({
  create: jest.fn(),
}));

const axios = require('axios');

const mockPost = jest.fn();
const mockGet = jest.fn();
const mockUseInterceptor = jest.fn();

axios.create.mockReturnValue({
  post: mockPost,
  get: mockGet,
  interceptors: {
    request: {
      use: mockUseInterceptor,
    },
  },
});

const { sendMessage, checkHealth } = require('./api');

describe('api service', () => {
  beforeEach(() => {
    mockPost.mockReset();
    mockGet.mockReset();
  });

  test('sendMessage posts chat payload and returns response data', async () => {
    const payload = { response: 'ok', session_id: 'abc' };
    mockPost.mockResolvedValueOnce({ data: payload });

    const result = await sendMessage('hello', 'abc');

    expect(mockPost).toHaveBeenCalledWith('/chat', {
      message: 'hello',
      session_id: 'abc',
    });
    expect(result).toEqual(payload);
  });

  test('checkHealth calls health endpoint and returns response data', async () => {
    const health = { status: 'healthy' };
    mockGet.mockResolvedValueOnce({ data: health });

    const result = await checkHealth();

    expect(mockGet).toHaveBeenCalledWith('/health');
    expect(result).toEqual(health);
  });

});
