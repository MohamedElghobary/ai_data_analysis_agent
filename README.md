# AI Data Analysis Agent 🤖📊

A powerful, intelligent data analysis tool that allows users to analyze CSV and Excel files using natural language queries. No SQL knowledge required!

## 🌟 Features

- **📁 File Upload Support**: Upload CSV and Excel files with automatic data type detection
- **🤖 Natural Language Queries**: Ask questions about your data in plain English
- **📊 Automatic Visualizations**: Generate charts and graphs with simple commands
- **📋 Statistical Analysis**: Get comprehensive data insights and summaries
- **🔍 Data Quality Assessment**: Identify data issues and patterns
- **⚙️ Interactive UI**: User-friendly Streamlit interface with real-time processing

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API Key (optional, for advanced AI features)

### Installation

1. **Clone or download the project**
```bash
git clone https://github.com/MohamedElghobary/ai_data_analysis_agent.git
cd ai_data_analysis_agent
```

2. **Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables (Optional)**

```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

5. **Run the application**

```bash
streamlit run main.py
```

