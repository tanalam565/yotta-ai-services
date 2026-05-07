/**
 * Application Entry Point
 * 
 * Bootstraps the React application and mounts the root App component
 * into the DOM element with id="root".
 * 
 * Uses React.StrictMode in development to highlight potential issues.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// ============ React App Bootstrap ============

/**
 * Create React root and render the main application component tree.
 */
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);