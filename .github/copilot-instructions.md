# AI Coding Agent Instructions - EDA Agentes LangChain

## Project Overview
This is an Exploratory Data Analysis (EDA) project using LangChain agents for intelligent data exploration and analysis. The project combines traditional data science workflows with AI-powered autonomous agents.

## Architecture Principles

### Core Components
- **Agents**: LangChain-based autonomous agents for data exploration
- **Tools**: Custom tools for data manipulation, visualization, and analysis
- **Data Pipeline**: ETL processes and data preparation workflows
- **Visualizations**: Charts, plots, and dashboards for insights presentation

### Agent Design Patterns
- Use LangChain's `AgentExecutor` for orchestrating multi-step analysis
- Implement custom tools using `@tool` decorator for domain-specific operations
- Structure agents with clear roles: DataExplorer, Visualizer, Statistician, ReportGenerator
- Maintain conversation memory for contextual analysis sessions

## File Organization

```
/
├── agents/          # LangChain agent definitions and configurations
├── tools/           # Custom tools for data analysis operations
├── data/           # Raw and processed datasets
│   ├── raw/        # Original, immutable data files
│   └── processed/  # Cleaned and transformed datasets
├── visualizations/ # Generated plots, charts, and dashboards
├── reports/        # Analysis reports and findings
└── utils/          # Helper functions and utilities
```

## Development Conventions

### Agent Implementation
- Name agents descriptively: `DataExplorerAgent`, `VisualizationAgent`
- Use type hints for all agent methods and tool functions
- Include docstrings with usage examples for custom tools
- Implement error handling for data quality issues and API failures

### Data Handling
- Always validate data integrity before analysis
- Use pandas for data manipulation, preserve original column names when possible

### Visualization Standards
- Use consistent color schemes across all visualizations
- Include proper titles, axis labels, and legends
- Save plots in both interactive (HTML) and static (PNG/SVG) formats
- Store visualization configurations for reproducibility

## Key Dependencies
- `langchain` - Core agent framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib`/`seaborn`/`plotly` - Visualization libraries


## Testing Approach
- Unit tests for custom tools using sample datasets
- Integration tests for agent workflows with known data scenarios
- Validation tests for data quality and analysis accuracy
- Performance tests for large dataset processing

## Common Patterns
- Use configuration files (YAML/JSON) for agent parameters and data sources
- Implement logging for agent decision tracking and debugging
- Create reusable tool chains for common analysis patterns
- Maintain separate environments for development and production analysis

## AI Agent Guidelines
- When creating agents, focus on specific analytical tasks rather than general-purpose functionality
- Implement proper error handling for common data issues (missing values, type mismatches)
- Use prompt engineering to guide agents toward relevant analysis techniques
- Document agent reasoning and decision-making processes in outputs
- Ensure agents can explain their analysis methods and conclusions clearly