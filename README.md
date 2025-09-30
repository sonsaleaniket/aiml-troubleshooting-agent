# AI/ML Troubleshooting Agent

A comprehensive Python-based troubleshooting agent that helps users resolve technical issues using a knowledge base and automated diagnostic procedures. This project combines natural language processing with machine learning to provide intelligent technical support that learns and improves over time.

## Features

### Basic Agent (`troubleshooting_agent.py`)
- **Intelligent Issue Matching**: Uses keyword matching to identify problems from user descriptions
- **Comprehensive Knowledge Base**: Pre-loaded with solutions for common technical issues
- **Interactive Diagnostics**: Guided troubleshooting procedures for network and performance issues
- **Automated Fixes**: Simulated automated solutions for common problems
- **User Feedback Collection**: Tracks solution effectiveness for continuous improvement
- **NLP Integration**: Uses Transformers library for advanced text processing capabilities

### Advanced Agent (`troubleshooting_agent_advanced.py`)
- **🧠 Machine Learning Integration**: Learns from every interaction to improve diagnostic accuracy
- **🔍 Semantic Similarity Matching**: Uses sentence transformers for better issue understanding
- **📊 Solution Effectiveness Prediction**: ML model predicts how likely solutions are to work
- **🔄 Adaptive Knowledge Base**: Continuously improves solutions based on user feedback
- **📈 Learning Analytics**: Tracks success rates and provides insights into agent performance
- **🎯 Confidence Scoring**: Provides confidence levels for issue matches and solutions
- **🔄 Continuous Learning**: Automatically retrains models as more data becomes available
- **🤖 Comprehensive Automation**: 13 automated fixes covering all major issue categories
- **📊 Automation Analytics**: Tracks automation success rates, resolution times, and user satisfaction
- **⚡ Performance Monitoring**: Real-time tracking of automation effectiveness and optimization

## Project Structure

```
aiml_troubleshooting/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── troubleshooting_agent.py           # Basic agent implementation
├── troubleshooting_agent_advanced.py  # Advanced ML-powered agent
├── troubleshooting_knowledge_base.json # Issue-solution database
├── interaction_data.json              # ML training data (auto-generated)
├── solution_effectiveness.json        # Solution effectiveness scores (auto-generated)
├── solution_predictor.pkl             # Trained ML model (auto-generated)
└── automation_analytics.json          # Automation performance data (auto-generated)
```

## Installation

1. **Clone or download this repository**
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Basic Agent

```bash
python troubleshooting_agent.py
```

### Running the Advanced ML Agent

```bash
python troubleshooting_agent_advanced.py
```


### Example Interaction (Advanced Agent)

```
🤖 Welcome to the Advanced Troubleshooting Agent!
🧠 This agent learns from every interaction to improve its accuracy.
============================================================

Please describe your problem (or 'quit' to exit, 'insights' for learning stats, 'automation' for automation analytics): My internet is very slow

🔧 Solution for 'slow_internet':
Try restarting your router and checking your connection settings.

📊 Confidence: High (85.2%)
🎯 Match Confidence: 90.5%

🔍 Running advanced network diagnostics...
Have you restarted your router?
Yes/No: no
🔍 Diagnostic Result: 🔄 Please restart your router and check your connection again. This resolves 70% of network issues.

🔄 Resetting network settings...
📡 Flushing DNS cache...
🔧 Resetting TCP/IP stack...
✅ Network settings have been reset. Please check your connection.
🔧 Advanced network settings reset completed.

📝 Enhanced Feedback Collection:
Did this solution resolve your issue? (Yes/No): yes
How would you rate the solution effectiveness? (1-10)
Rating: 8
✅ Great! Your feedback has been recorded and will improve future recommendations.
🔄 Updating ML models with new feedback...
```

## Supported Issues

The knowledge base includes solutions for:

- **Network Issues**: Slow internet, WiFi connection problems
- **System Performance**: Slow computer, frozen system
- **Hardware Problems**: Printer issues, external drive recognition
- **Software Issues**: App crashes, browser performance, file access
- **System Errors**: Blue screen errors, sound problems
- **Account Issues**: Password recovery, email sending problems

## Key Components

### TroubleshootingAgent Class

The main class that handles:
- Knowledge base loading and management
- Issue matching and solution provision
- Interactive diagnostic procedures
- Automated fix attempts
- User feedback collection

### Knowledge Base

The `troubleshooting_knowledge_base.json` file contains:
- Issue identifiers
- Symptom descriptions
- Step-by-step solutions
- Categorized problem types

### Diagnostic Procedures

- **Network Diagnostics**: Router restart, cable checks, ISP contact
- **Performance Diagnostics**: Program count analysis, disk space checks

## Customization

### Adding New Issues

To add new troubleshooting scenarios, edit `troubleshooting_knowledge_base.json`:

```json
{
    "new_issue_key": {
        "symptom": "Description of the problem",
        "solution": "Step-by-step solution instructions"
    }
}
```

### Extending Diagnostics

Add new diagnostic procedures by implementing methods in the `TroubleshootingAgent` class:

```python
def _diagnose_new_issue_type(self) -> str:
    """Custom diagnostic procedure"""
    # Implementation here
    return "Diagnostic result"
```

## Dependencies

### Basic Agent
- **transformers**: For NLP model integration and question-answering capabilities
- **spacy**: For advanced natural language processing (optional)

### Advanced Agent (Additional Dependencies)
- **scikit-learn**: For machine learning models and algorithms
- **numpy**: For numerical computations
- **pandas**: For data manipulation and analysis
- **sentence-transformers**: For semantic similarity matching
- **torch**: For deep learning capabilities
- **joblib**: For model persistence and loading

## Error Handling

The agent includes comprehensive error handling for:
- Missing knowledge base files
- Malformed JSON data
- NLP model initialization failures
- User input validation
- Keyboard interrupts

## Machine Learning Features

The advanced agent includes several ML-powered features:

### 1. Semantic Similarity Matching
- Uses sentence transformers to understand user intent beyond keyword matching
- Provides confidence scores for issue matches
- Handles variations in problem descriptions

### 2. Solution Effectiveness Prediction
- Random Forest classifier trained on user feedback
- Predicts likelihood of solution success
- Continuously improves with more interaction data

### 3. Adaptive Knowledge Base
- Tracks effectiveness scores for each solution
- Suggests improved solutions based on learning
- Maintains rolling averages of solution success rates

### 4. Continuous Learning
- Automatically retrains models every 5 interactions
- Stores interaction data for future model improvements
- Provides learning insights and analytics

## Automation Features

The advanced agent includes comprehensive automation capabilities:

### 1. System-Level Automation
- **Blue Screen Error Resolution**: Comprehensive diagnostics including memory checks, driver updates, malware scanning
- **System Optimization**: Complete system cleanup, startup optimization, disk defragmentation, antivirus updates
- **Frozen Computer Recovery**: Resource conflict detection and process termination

### 2. Network Automation
- **Advanced Network Diagnostics**: DNS cache flushing, TCP/IP stack reset, IP configuration renewal
- **WiFi Troubleshooting**: Adapter reset, driver updates, signal strength analysis, configuration optimization

### 3. Software Automation
- **Browser Optimization**: Comprehensive cleanup including cache, cookies, history, extensions, and security updates
- **App Crash Resolution**: Log analysis, version updates, cache clearing, compatibility checks
- **File Access Troubleshooting**: Permission verification, association repair, corruption scanning, antivirus checks

### 4. Hardware Automation
- **Printer Troubleshooting**: Connectivity diagnostics, driver updates, network testing, spooler service restart
- **External Drive Fixes**: USB port testing, driver updates, error scanning, device manager refresh
- **Audio System Repair**: Device status checks, driver updates, service restart, permission verification

### 5. Email & Security Automation
- **Email Configuration**: Server settings verification, SMTP/POP3 testing, authentication checks
- **Password Recovery**: Account recovery analysis, security question verification, reset link generation

### 6. Automation Analytics
- **Performance Tracking**: Success rates, resolution times, user satisfaction scores
- **ML Integration**: Learning from automation attempts to improve future fixes
- **Real-time Monitoring**: Continuous tracking of automation effectiveness

## Future Enhancements

Potential improvements for this project:

1. **Deep Learning Models**: Implement neural networks for more sophisticated understanding
2. **Web Interface**: Create a web-based UI for easier access
3. **Database Integration**: Replace JSON with a proper database
4. **Multi-language Support**: Add support for multiple languages
5. **Advanced NLP**: Implement more sophisticated text understanding
6. **Logging System**: Add comprehensive logging for debugging
7. **API Development**: Create REST API for integration with other systems
8. **Real-time Learning**: Implement online learning algorithms
9. **Ensemble Methods**: Combine multiple ML models for better accuracy
10. **User Profiling**: Learn user-specific patterns and preferences

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues or questions about this troubleshooting agent, please create an issue in the repository or contact the development team.

---

**Note**: This is a learning project demonstrating AI/ML concepts in troubleshooting applications. The automated fixes are simulated and should not be used in production environments without proper testing and validation.
