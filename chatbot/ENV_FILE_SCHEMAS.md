------------------------------------------------------Backend .env--------------------------------------------

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=
AZURE_STORAGE_CONTAINER_NAME=
AZURE_UPLOADS_CONTAINER_NAME=user-uploads

# Azure AI Search
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_KEY=
AZURE_SEARCH_INDEX_NAME=

# AZURE_SEARCH_INDEX_NAME=yotta-property-docs
AZURE_SEARCH_DATASOURCE_NAME=
AZURE_SEARCH_INDEXER_NAME

# Redis
REDIS_URL=rediss://default:your-redis-access-key@your-cache-name.redis.cache.windows.net:6380/0

# Azure OpenAI
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT_NAME=
AZURE_OPENAI_API_VERSION=

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=
AZURE_DOCUMENT_INTELLIGENCE_KEY=

# Durable source-of-truth storage (chat + uploads metadata/pages)
DATABASE_URL=postgresql://user:password@host:5432/yottareal
PERSISTENCE_DB_PATH=backend/data/chat_logs.db

# Azure OpenAI Embeddings
AZURE_OPENAI_EMBEDDING_ENDPOINT=
AZURE_OPENAI_EMBEDDING_KEY=
AZURE_OPENAI_EMBEDDING_
AZURE_OPENAI_EMBEDDING_MODEL=
AZURE_OPENAI_EMBEDDING_API_VERSION=

# API Authentication
CHATBOT_API_KEY=your-secret-api-key-here

----------------------------------------------------Frontend .env-----------------------------------------------------

# Backend API Configuration
REACT_APP_API_URL=http://localhost:8000/api

# API Authentication (must match backend CHATBOT_API_KEY)
REACT_APP_CHATBOT_API_KEY=your-secret-api-key-here

                                   ==============Production Frontend .env==============
# Backend API Configuration
REACT_APP_API_URL=https://your-production-server.com/api

# API Authentication
REACT_APP_CHATBOT_API_KEY=your-secret-api-key-here

************************************************* RUN FRONTEND ***************************************************
cd frontend
npm install
npm start