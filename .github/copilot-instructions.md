# AI Coding Agent Instructions - EDA Agentes LangChain

## Project Overview
This is an Exploratory Data Analysis (EDA) project using LangChain agents integrated with Streamlit for intelligent data exploration and analysis. The project creates a conversational interface where users can ask natural language questions about CSV datasets and receive comprehensive analyses with visualizations.

## Architecture Principles

### Core Components
- **Streamlit Interface**: Web-based user interface for file upload and chat interaction
- **LangChain Agents**: AI-powered agents for autonomous data analysis using Google's Gemini Pro
- **Custom Tools**: Specialized tools for data analysis, visualization, and statistical operations
- **Memory System**: Conversational memory to maintain context across analysis sessions
- **Visualization Engine**: Automatic generation of charts and plots using Plotly, Seaborn, and Matplotlib

### Agent Design Patterns
- Use LangChain's `AgentExecutor` with Google Gemini Pro for orchestrating multi-step analysis
- Implement custom tools using `@tool` decorator for domain-specific EDA operations
- Structure agents with clear roles: EDAAgent (main), DataExplorer, Statistician, Visualizer
- Maintain conversation memory using `ConversationBufferWindowMemory` for contextual analysis sessions
- Use environment variables (.env) for configuration management
- Always use the `eda_lang` conda environment for consistency

## File Organization

```
/
├── .env                          # Environment configuration (API keys, settings)
├── .env.example                  # Template for environment variables
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment specification
├── app.py                        # Main Streamlit application
├── IMPLEMENTATION_GUIDE.md       # Detailed implementation guide
├── config/
│   └── settings.py              # Centralized configuration management
├── agents/          # LangChain agent definitions and configurations
│   ├── eda_agent.py             # Main EDA agent with memory
│   ├── data_explorer_agent.py   # Data exploration specialist
│   ├── statistician_agent.py    # Statistical analysis specialist
│   └── visualizer_agent.py      # Visualization specialist
├── tools/           # Custom tools for data analysis operations
│   ├── data_analysis_tools.py   # Data description and type analysis
│   ├── visualization_tools.py   # Chart and plot generation
│   ├── statistical_tools.py     # Statistical computations
│   └── outlier_detection_tools.py # Anomaly detection
├── data/           # Data storage and caching
│   ├── uploads/    # User-uploaded CSV files
│   └── cache/      # Cached analysis results
├── utils/          # Helper functions and utilities
│   ├── data_loader.py           # CSV loading and validation
│   ├── memory_manager.py        # Conversation memory management
│   └── visualization_helpers.py # Chart formatting utilities
└── tests/          # Test suite
    ├── test_agents.py           # Agent functionality tests
    ├── test_tools.py            # Tool functionality tests
    └── test_data/               # Sample datasets for testing
        └── sample.csv
```

## Development Conventions

### Agent Implementation
- Name agents descriptively: `EDAAgent`, `DataExplorerAgent`, `VisualizationAgent`
- Use type hints for all agent methods and tool functions
- Include docstrings with usage examples for custom tools
- Implement error handling for data quality issues and API failures
- Always use Google Gemini Pro as the LLM provider via `langchain-google-genai`
- Configure agents with proper memory management for conversational context

### Data Handling
- Always validate data integrity before analysis
- Use pandas for data manipulation, preserve original column names when possible
- Implement proper CSV loading with encoding detection
- Handle missing values and data type conversions gracefully
- Cache analysis results to improve performance

### Visualization Standards
- Use Plotly as primary visualization library for interactive charts
- Use consistent color schemes across all visualizations
- Include proper titles, axis labels, and legends
- Generate both interactive (Plotly) and static visualizations when needed
- Embed visualizations directly in Streamlit interface
- Store visualization configurations for reproducibility

### Streamlit Integration
- Use `st.file_uploader` for CSV file uploads
- Implement chat interface with `st.chat_message` and `st.chat_input`
- Display analysis results with proper formatting and visualizations
- Maintain session state for conversation history and uploaded data
- Use loading spinners for long-running analysis operations

## Key Dependencies
- `streamlit>=1.28.0` - Web interface framework
- `langchain>=0.0.350` - Core agent framework
- `langchain-google-genai>=0.0.5` - Google Gemini Pro integration
- `pandas>=2.1.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computations
- `plotly>=5.17.0` - Interactive visualizations
- `seaborn>=0.12.0` - Statistical visualizations
- `matplotlib>=3.7.0` - Base plotting library
- `scipy>=1.11.0` - Statistical functions
- `scikit-learn>=1.3.0` - Machine learning utilities
- `python-dotenv>=1.0.0` - Environment variable management


## Testing Approach
- Unit tests for custom tools using sample datasets in `tests/test_data/`
- Integration tests for agent workflows with known data scenarios
- Validation tests for data quality and analysis accuracy
- Performance tests for large dataset processing (up to 100MB CSV files)
- Test Streamlit interface components and user interactions
- Automated testing with pytest framework

## Environment Management
- Always use the `eda_lang` conda environment for development and testing
- Manage dependencies through both `environment.yml` (conda) and `requirements.txt` (pip)
- Use `.env` file for configuration management (API keys, settings)
- Maintain separate configurations for development and production

## Common Patterns
- Use `.env` files for environment configuration and API key management
- Implement logging for agent decision tracking and debugging
- Create reusable tool chains for common EDA analysis patterns
- Cache analysis results to improve performance and user experience
- Maintain conversation history in Streamlit session state
- Use Streamlit's caching mechanisms for expensive operations

## AI Agent Guidelines
- When creating agents, focus on specific EDA tasks: data description, visualization, statistical analysis, outlier detection
- Implement proper error handling for common data issues (missing values, type mismatches, encoding problems)
- Use prompt engineering to guide agents toward relevant analysis techniques for CSV datasets
- Always generate visualizations to support analysis conclusions
- Maintain conversational context using LangChain's memory components
- Ensure agents can explain their analysis methods and provide actionable insights
- Design agents to ask clarifying questions when data analysis requirements are ambiguous
- Implement tools that can handle various CSV formats and data types
- Provide conclusions and recommendations based on the analysis performed