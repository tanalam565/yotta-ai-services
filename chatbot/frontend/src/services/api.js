/**
 * API Service Layer
 * 
 * Centralizes all HTTP communication between the frontend and backend API.
 * Provides preconfigured Axios client, API-key headers, and helper
 * functions for upload, chat, cleanup, and health endpoints.
 */

import axios from 'axios';

// ============ Configuration Constants ============

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
const API_KEY = process.env.REACT_APP_CHATBOT_API_KEY || '';

/**
 * Preconfigured Axios instance for backend API calls.
 */
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((requestConfig) => {
  if (API_KEY) {
    requestConfig.headers['X-API-Key'] = API_KEY;
  }
  return requestConfig;
});

export const uploadDocument = async (file, sessionId) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('session_id', sessionId);
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const cleanupSession = async (sessionId) => {
  const response = await api.post('/cleanup-session', { session_id: sessionId });
  return response.data;
};

/**
 * Send a chat message to the backend chat endpoint.
 * 
 * @async
 * @param {string} messageText - User message text to send.
 * @param {string|null} [sessionId=null] - Optional session id for continuity.
 * @returns {Promise<Object>} Chat response payload containing response text, sources, and session id.
 */
export const sendMessage = async (messageText, sessionId = null) => {
  try {
    const response = await api.post('/chat', {
      message: messageText,
      session_id: sessionId,
    });
    return response.data;
  } catch (error) {
    console.error('Error sending message:', error);
    throw error;
  }
};

/**
 * Check backend API health status.
 * 
 * @async
 * @returns {Promise<Object>} Health status object from backend.
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Error checking health:', error);
    throw error;
  }
};

/**
 * Default export for direct API usage when needed.
 */
export default api;