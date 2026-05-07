/**
 * App Component - Main Application Root
 * 
 * The root component of the YottaReal Bot application. Provides the overall
 * layout structure including:
 * - Header with application branding and tagline
 * - Main content area with the ChatInterface component
 * - Footer with copyright information
 * 
 * This component serves as the entry point for the entire application and
 * establishes the visual layout and structure for all child components.
 */

import React from 'react';
import ChatInterface from './components/ChatInterface';
import './App.css';

/**
 * App - Root React component for the YottaReal chatbot application
 * 
 * @component
 * @returns {React.ReactElement} The main application layout with header, chat interface, and footer
 */
function App() {
  const currentYear = new Date().getFullYear();
  
  // ============ Application Layout ============

  return (
    <div className="app">
      {/* Header Section - Application Branding */}
      <header className="site-header">
        <div className="wordmark">
          {/* Application Logo/Name */}
          <div className="name">
            {/* Primary brand: "YOTTA" */}
            <span className="yotta">YOTTA</span>
            {/* Secondary brand: "REAL" */}
            <span className="real">REAL</span>
          </div>
          {/* Application tagline/description */}
          <div className="tagline">SMART PARTNER</div>
        </div>
      </header>
      
      {/* Main Content Area - Chat Interface */}
      <main className="chat-main">
        <ChatInterface />
      </main>
      
      {/* Footer Section - Copyright Information */}
      <footer className="site-footer">
        {/* Copyright notice with dynamic year */}
        <small className="copyright">© {currentYear} YottaReal</small>
      </footer>
    </div>
  );
}

// ============ Export ============

/**
 * Export the App component as the default export for use in index.js
 * 
 * The App component is the root component of the React application tree.
 * It should be mounted to the DOM in the main index.js entry point.
 */
export default App;