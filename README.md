# Medical Report Analyzer

An AI-powered medical coding assistant that extracts entities from medical reports and retrieves relevant ICD-11, CPT/HCPCS codes, comorbidities, and Z-codes using a microservices architecture with MCP (Model Context Protocol) server and Streamlit interface.

## Features

- **Entity Extraction**: Automatically identifies conditions, procedures, and health factors from medical reports
- **Interactive Visualization**: Click highlighted entities in reports to view related medical codes
- **Comprehensive Code Retrieval**:
  - ICD-11 diagnostic codes with WHO API integration
  - CPT/HCPCS procedure codes
  - Comorbidity analysis using FAISS vector search
  - Z-codes for health status factors
- **AI-Powered Explanations**: Gemini-generated explanations for each medical code
- **Multi-Format Support**: Process PDF documents and medical images (JPG, PNG)
- **OCR Integration**: Extract text from scanned reports using Tesseract

## Architecture

The system uses a microservices architecture with two Docker containers:

1. **MCP Server** (`server:8000`): Handles medical coding logic, vector database queries, and API integrations
2. **Streamlit Agent** (`agent:8501`): Provides the web interface for uploading and analyzing reports

## Prerequisites

- Docker and Docker Compose
- Google Gemini API key
- WHO ICD-11 API credentials (Client ID and Secret)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone [<repository-url>](https://github.com/ib105/Agentic-Healthcare)
cd Agentic-Healthcare
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with the following credentials:

```env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
ICD_CLIENT_ID=your_icd_client_id_here
ICD_CLIENT_SECRET=your_icd_client_secret_here
MCP_SERVER_URL=http://server:8000
```

**Obtaining API Keys:**

- **Gemini API**: Get your key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **ICD-11 API**: Register at [WHO ICD API](https://icd.who.int/icdapi) to obtain client credentials

### 3. Prepare Vector Databases

Before building the Docker containers, you need to create the FAISS vector databases:

#### Comorbidity Database

```bash
# Ensure you have Comorbidities.xlsx in the project root
python comorbidity_data_prep.py
```

This creates `./comorbidities_faiss/` directory with the vector store.

#### CPT Code Database

```bash
# Ensure you have the CPT Excel file in CPT/ directory
python cpt_data_prep.py
```

This creates `CPT/cpt_faiss/` directory with the vector store.

### 4. Build and Run with Docker Compose

```bash
docker-compose up --build
```

This command will:
- Build both server and agent containers
- Create a shared volume for file uploads
- Set up networking between containers
- Start health checks to ensure proper startup

### 5. Access the Application

Once the containers are running:

- **Streamlit Interface**: http://localhost:8501
- **MCP Server API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (FastAPI auto-generated)

## Usage

1. **Upload a Medical Report**
   - Navigate to http://localhost:8501
   - Click "Upload Report" and select a PDF or image file
   - Supported formats: PDF, PNG, JPG, JPEG

2. **View Extracted Entities**
   - The system automatically extracts conditions, procedures, and health factors
   - Entities are color-coded:
     - 🔴 Red: Conditions/Diagnoses
     - 🔵 Blue: Procedures/Services
     - 🟢 Green: Health Status Factors

3. **Explore Medical Codes**
   - Click any highlighted entity in the report
   - Browse comorbidities, ICD-11 codes, CPT codes, or Z-codes
   - Click individual codes for AI-generated explanations

## Project Structure

```
.
├── docker-compose.yml          # Container orchestration
├── Dockerfile.agent            # Streamlit app container
├── Dockerfile.server           # MCP server container
├── .dockerignore              # Docker build exclusions
├── .env                       # Environment variables (create this)
├── pyproject.toml             # Python dependencies
├── uv.lock                    # Locked dependencies
│
├── app.py                     # Streamlit web interface
├── client.py                  # MCP client for Gemini integration
├── server.py                  # FastAPI MCP server
├── tools.py                   # Tool implementations
│
├── report_loading.py          # PDF/image text extraction
├── comorbidity_retriever.py   # FAISS comorbidity search
├── cpt_retriever.py           # CPT code search
├── icd11_retriever.py         # WHO ICD-11 API integration
│
├── comorbidity_data_prep.py   # Build comorbidity vector DB
├── cpt_data_prep.py           # Build CPT vector DB
├── main.py                    # LangGraph pipeline (optional)
│
├── comorbidities_faiss/       # Comorbidity vector store
├── CPT/                       # CPT codes and vector store
└── Med-reports/               # Sample reports (not in Docker)
```

## Development

### Running Locally (Without Docker)

If you want to run the services locally for development:

1. Install dependencies with uv:
```bash
pip install uv
uv sync
```

2. Start the MCP server:
```bash
uv run python server.py
```

3. In a separate terminal, start Streamlit:
```bash
uv run streamlit run app.py
```

### Testing the MCP Client

Test Gemini integration with MCP tools:

```bash
uv run python client.py
```

### Running the LangGraph Pipeline

For batch processing of reports:

```bash
uv run python main.py
```

## API Endpoints

The MCP server exposes the following REST API endpoints:

- `GET /` - Server status
- `GET /health` - Health check
- `GET /tools` - List available MCP tools
- `POST /call-tool` - Execute MCP tool
- `POST /upload` - Upload medical report file

### Available MCP Tools

1. `extract_report_text` - Extract text from PDF/image
2. `extract_medical_entities` - NLP entity extraction
3. `find_comorbidities` - Search related conditions
4. `find_icd11_codes` - Search ICD-11 diagnostic codes
5. `find_cpt_codes` - Search CPT procedure codes
6. `find_z_codes` - Search health status Z-codes
7. `generate_explanation` - AI explanation for a single code
8. `generate_batch_explanations` - Batch explanations (quota-optimized)

## Troubleshooting

### Server Connection Issues

If the agent can't connect to the server:
```bash
# Check server logs
docker logs medical-mcp-server

# Verify network
docker network inspect medical-network
```

### OCR Not Working

Ensure Tesseract is installed in containers (already configured in Dockerfiles):
```dockerfile
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev
```

### API Quota Errors

If you hit Gemini API limits:
- Use `generate_batch_explanations` instead of multiple `generate_explanation` calls
- Reduce the number of codes retrieved (adjust `top_k` and `limit` parameters)
- Implement rate limiting or caching

### Vector Database Not Found

Ensure you ran the data preparation scripts before building:
```bash
python comorbidity_data_prep.py
python cpt_data_prep.py
```

## Performance Optimization

- **Caching**: Streamlit caches OCR results and code explanations
- **Batch Processing**: Batch explanations reduce API calls from N to 1
- **Parallel Execution**: Multiple code searches run concurrently
- **Vector Search**: FAISS provides fast similarity search

## Security Considerations

- Keep API keys in `.env` file (never commit to Git)
- The `.dockerignore` excludes sensitive files from images
- Vector databases use `allow_dangerous_deserialization` (only safe in trusted environments)
- File uploads are isolated in Docker volumes

## Support

For issues and questions:
- Check the [API documentation](http://localhost:8000/docs) when running
- Review Docker logs: `docker-compose logs -f`
- Verify environment variables in `.env`

---

**Note**: This is a prototype system.
