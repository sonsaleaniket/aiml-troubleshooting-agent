# AI/ML Troubleshooting Agent

A comprehensive Python-based troubleshooting agent that helps users resolve technical issues using a knowledge base and automated diagnostic procedures. This project combines natural language processing with interactive problem-solving to provide intelligent technical support.

## Features

- **Intelligent Issue Matching**: Uses keyword matching to identify problems from user descriptions
- **Comprehensive Knowledge Base**: Pre-loaded with solutions for common technical issues
- **Interactive Diagnostics**: Guided troubleshooting procedures for network and performance issues
- **Automated Fixes**: Simulated automated solutions for common problems
- **User Feedback Collection**: Tracks solution effectiveness for continuous improvement
- **NLP Integration**: Uses Transformers library for advanced text processing capabilities

## Project Structure

```
aiml_troubleshooting/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── troubleshooting_agent.py           # Main agent implementation
└── troubleshooting_knowledge_base.json # Issue-solution database
```

## Installation

1. **Clone or download this repository**
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Agent

```bash
python troubleshooting_agent.py
```

### Example Interaction

```
🤖 Welcome to the Troubleshooting Agent!
==================================================

Please describe your problem (or 'quit' to exit): My internet is very slow

🔧 Solution for 'slow_internet':
Try restarting your router and checking your connection settings.

🔍 Running network diagnostics...
Have you restarted your router?
Yes/No: no
🔍 Diagnostic Result: Please restart your router and check your connection again.

🔄 Resetting network settings...
✅ Network settings have been reset. Please check your connection.
🔧 Network settings reset completed.

📝 Feedback Collection:
Did this solution resolve your issue? (Yes/No): yes
✅ Great! Your feedback has been recorded.
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

- **transformers**: For NLP model integration and question-answering capabilities
- **spacy**: For advanced natural language processing (optional)

## Error Handling

The agent includes comprehensive error handling for:
- Missing knowledge base files
- Malformed JSON data
- NLP model initialization failures
- User input validation
- Keyboard interrupts

## Future Enhancements

Potential improvements for this project:

1. **Machine Learning Integration**: Train custom models on troubleshooting data
2. **Web Interface**: Create a web-based UI for easier access
3. **Database Integration**: Replace JSON with a proper database
4. **Multi-language Support**: Add support for multiple languages
5. **Advanced NLP**: Implement more sophisticated text understanding
6. **Logging System**: Add comprehensive logging for debugging
7. **API Development**: Create REST API for integration with other systems

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
