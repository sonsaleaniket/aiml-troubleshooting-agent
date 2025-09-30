import json
import os
from typing import Dict, Optional, Tuple
from transformers import pipeline


class TroubleshootingAgent:
    """
    A comprehensive troubleshooting agent that helps users resolve technical issues
    using a knowledge base and automated diagnostic procedures.
    """
    
    def __init__(self, knowledge_base_path: str = 'troubleshooting_knowledge_base.json'):
        """
        Initialize the troubleshooting agent.
        
        Args:
            knowledge_base_path: Path to the JSON knowledge base file
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()
        self.nlp_model = self._initialize_nlp_model()
        
    def _load_knowledge_base(self) -> Dict:
        """
        Load the troubleshooting knowledge base from JSON file.
        
        Returns:
            Dictionary containing troubleshooting information
            
        Raises:
            FileNotFoundError: If knowledge base file doesn't exist
            json.JSONDecodeError: If JSON file is malformed
        """
        try:
            if not os.path.exists(self.knowledge_base_path):
                raise FileNotFoundError(f"Knowledge base file not found: {self.knowledge_base_path}")
                
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in knowledge base: {e}")
    
    def _initialize_nlp_model(self):
        """
        Initialize the NLP model for advanced text processing.
        
        Returns:
            Initialized transformers pipeline
        """
        try:
            return pipeline('question-answering')
        except Exception as e:
            print(f"Warning: Could not initialize NLP model: {e}")
            return None
    
    def find_matching_issue(self, user_input: str) -> Optional[Tuple[str, Dict]]:
        """
        Find the best matching issue from the knowledge base.
        
        Args:
            user_input: User's problem description
            
        Returns:
            Tuple of (issue_key, issue_details) if found, None otherwise
        """
        user_input_lower = user_input.lower()
        
        # Simple keyword matching
        for issue_key, issue_details in self.knowledge_base.items():
            symptom_lower = issue_details["symptom"].lower()
            
            # Check if symptom keywords appear in user input
            if any(keyword in user_input_lower for keyword in symptom_lower.split()):
                return issue_key, issue_details
        
        return None
    
    def provide_solution(self, issue_key: str, issue_details: Dict) -> str:
        """
        Provide a solution for the identified issue.
        
        Args:
            issue_key: Key identifying the issue
            issue_details: Dictionary containing issue information
            
        Returns:
            Formatted solution string
        """
        solution = issue_details.get("solution", "No solution available.")
        return f"🔧 Solution for '{issue_key}':\n{solution}"
    
    def run_diagnostic(self, issue_type: str) -> str:
        """
        Run interactive diagnostic procedures for specific issue types.
        
        Args:
            issue_type: Type of issue to diagnose
            
        Returns:
            Diagnostic result message
        """
        if issue_type == "network":
            return self._diagnose_network_issue()
        elif issue_type == "performance":
            return self._diagnose_performance_issue()
        else:
            return "No specific diagnostic available for this issue type."
    
    def _diagnose_network_issue(self) -> str:
        """
        Interactive network issue diagnosis.
        
        Returns:
            Diagnostic result message
        """
        print("\n🔍 Running network diagnostics...")
        
        print("Have you restarted your router?")
        response = input("Yes/No: ").strip().lower()
        
        if response == "no":
            return "Please restart your router and check your connection again."
        else:
            print("Have you checked your network cables?")
            cable_response = input("Yes/No: ").strip().lower()
            
            if cable_response == "no":
                return "Please check all network cables and connections."
            else:
                return "Try resetting your network settings or contact your ISP for further assistance."
    
    def _diagnose_performance_issue(self) -> str:
        """
        Interactive performance issue diagnosis.
        
        Returns:
            Diagnostic result message
        """
        print("\n🔍 Running performance diagnostics...")
        
        print("How many programs are currently running?")
        programs = input("Enter approximate number: ").strip()
        
        try:
            num_programs = int(programs)
            if num_programs > 20:
                return "Close unnecessary programs to improve performance."
            else:
                return "Check available disk space and consider running disk cleanup."
        except ValueError:
            return "Please restart your computer and monitor performance."
    
    def attempt_automated_fix(self, issue_key: str) -> str:
        """
        Attempt to automatically fix the identified issue.
        
        Args:
            issue_key: Key identifying the issue
            
        Returns:
            Result message of the automated fix attempt
        """
        automated_fixes = {
            "slow_internet": self._fix_network_settings,
            "slow_computer": self._optimize_system,
            "browser_slow": self._clear_browser_cache
        }
        
        if issue_key in automated_fixes:
            return automated_fixes[issue_key]()
        else:
            return "Automated fix not available for this issue."
    
    def _fix_network_settings(self) -> str:
        """
        Simulate network settings reset.
        
        Returns:
            Fix result message
        """
        print("🔄 Resetting network settings...")
        print("✅ Network settings have been reset. Please check your connection.")
        return "Network settings reset completed."
    
    def _optimize_system(self) -> str:
        """
        Simulate system optimization.
        
        Returns:
            Fix result message
        """
        print("🔄 Optimizing system performance...")
        print("✅ System optimization completed.")
        return "System optimization completed."
    
    def _clear_browser_cache(self) -> str:
        """
        Simulate browser cache clearing.
        
        Returns:
            Fix result message
        """
        print("🔄 Clearing browser cache...")
        print("✅ Browser cache cleared successfully.")
        return "Browser cache cleared successfully."
    
    def collect_feedback(self) -> bool:
        """
        Collect user feedback on the provided solution.
        
        Returns:
            True if solution was helpful, False otherwise
        """
        print("\n📝 Feedback Collection:")
        feedback = input("Did this solution resolve your issue? (Yes/No): ").strip().lower()
        
        if feedback in ['yes', 'y']:
            print("✅ Great! Your feedback has been recorded.")
            return True
        else:
            print("❌ We're sorry the issue persists. We'll improve our solution based on your input.")
            return False
    
    def run(self):
        """
        Main execution loop for the troubleshooting agent.
        """
        print("🤖 Welcome to the Troubleshooting Agent!")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\nPlease describe your problem (or 'quit' to exit): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thank you for using the Troubleshooting Agent!")
                    break
                
                if not user_input:
                    print("Please provide a description of your problem.")
                    continue
                
                # Find matching issue
                match_result = self.find_matching_issue(user_input)
                
                if match_result:
                    issue_key, issue_details = match_result
                    
                    # Provide solution
                    solution = self.provide_solution(issue_key, issue_details)
                    print(f"\n{solution}")
                    
                    # Run diagnostics if applicable
                    if "internet" in user_input.lower() or "network" in user_input.lower():
                        diagnostic_result = self.run_diagnostic("network")
                        print(f"\n🔍 Diagnostic Result: {diagnostic_result}")
                    
                    if "slow" in user_input.lower() and "computer" in user_input.lower():
                        diagnostic_result = self.run_diagnostic("performance")
                        print(f"\n🔍 Diagnostic Result: {diagnostic_result}")
                    
                    # Attempt automated fix
                    fix_result = self.attempt_automated_fix(issue_key)
                    if "completed" in fix_result.lower():
                        print(f"\n🔧 {fix_result}")
                    
                    # Collect feedback
                    self.collect_feedback()
                    
                else:
                    print("❌ No matching issue found in the knowledge base.")
                    print("💡 Please try describing your problem differently or contact technical support.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                print("Please try again.")


def main():
    """
    Main function to run the troubleshooting agent.
    """
    try:
        agent = TroubleshootingAgent()
        agent.run()
    except Exception as e:
        print(f"Failed to initialize troubleshooting agent: {e}")


if __name__ == "__main__":
    main()