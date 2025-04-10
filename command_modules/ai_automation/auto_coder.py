#!/usr/bin/env python3

import os
import sys
import logging
import subprocess
import json
import re
import ast
import inspect
import tempfile
import shutil
import git
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from datetime import datetime
import importlib.util
import traceback
import black
import pylint.lint
import mypy.api
import autopep8
import pyflakes.api
import bandit.core.manager
import safety.cli
import requests
import openai
from jinja2 import Environment, FileSystemLoader

try:
    import langchain
    import transformers
    import torch
except ImportError:
    print("Installing required dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", 
                   "langchain", "transformers", "torch"])
    import langchain
    import transformers
    import torch

class AutoCoder:
    def __init__(self, config_file: str = "config/auto_coder_config.json"):
        """Initialize the AutoCoder with configuration."""
        self.config_file = Path(config_file)
        self.config = self.load_config()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize AI models
        self.initialize_models()
        
        # Initialize code templates
        self.initialize_templates()
        
        # Initialize code analysis tools
        self.initialize_analysis_tools()
        
        # Initialize results tracking
        self.generation_results = {
            "code_snippets": [],
            "files_created": [],
            "errors": [],
            "warnings": [],
            "security_issues": []
        }

    def load_config(self) -> Dict:
        """Load auto coder configuration from file."""
        default_config = {
            "languages": {
                "python": {
                    "enabled": True,
                    "style_guide": "pep8",
                    "formatter": "black"
                },
                "javascript": {
                    "enabled": True,
                    "style_guide": "airbnb",
                    "formatter": "prettier"
                },
                "ruby": {
                    "enabled": True,
                    "style_guide": "rubocop",
                    "formatter": "rubocop"
                },
                "c": {
                    "enabled": True,
                    "style_guide": "gnu",
                    "formatter": "clang-format"
                },
                "go": {
                    "enabled": True,
                    "style_guide": "gofmt",
                    "formatter": "gofmt"
                },
                "shell": {
                    "enabled": True,
                    "style_guide": "shellcheck",
                    "formatter": "shfmt"
                },
                "html": {
                    "enabled": True,
                    "style_guide": "html5",
                    "formatter": "prettier"
                }
            },
            "ai": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000
            },
            "security": {
                "audit_enabled": True,
                "vulnerability_check": True,
                "static_analysis": True
            },
            "testing": {
                "auto_test": True,
                "coverage": True,
                "debug": True
            }
        }
        
        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                return json.load(f)
        else:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(default_config, f, indent=4)
            return default_config

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs/auto_coder")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"auto_coder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_models(self):
        """Initialize AI models for code generation."""
        try:
            # Initialize OpenAI API
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                self.logger.warning("OPENAI_API_KEY not set. Some features may be limited.")
            
            # Initialize language models
            self.models = {
                "code_generation": "gpt-4",
                "code_analysis": "gpt-4",
                "security_audit": "gpt-4"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI models: {e}")

    def initialize_templates(self):
        """Initialize code templates and patterns."""
        self.templates = {
            "python": {
                "web_app": self._load_template("python_web_app.j2"),
                "api": self._load_template("python_api.j2"),
                "script": self._load_template("python_script.j2"),
                "exploit": self._load_template("python_exploit.j2"),
                "network": self._load_template("python_network.j2")
            },
            "javascript": {
                "web_app": self._load_template("javascript_web_app.j2"),
                "api": self._load_template("javascript_api.j2"),
                "script": self._load_template("javascript_script.j2")
            },
            "html": {
                "form": self._load_template("html_form.j2"),
                "page": self._load_template("html_page.j2"),
                "responsive": self._load_template("html_responsive.j2")
            },
            "shell": {
                "script": self._load_template("shell_script.j2"),
                "automation": self._load_template("shell_automation.j2")
            }
        }

    def _load_template(self, template_name: str) -> Any:
        """Load a Jinja2 template."""
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True
        )
        
        return env.get_template(template_name)

    def initialize_analysis_tools(self):
        """Initialize code analysis and security tools."""
        self.analysis_tools = {
            "python": {
                "linter": pylint.lint,
                "type_checker": mypy.api,
                "formatter": black,
                "security": bandit.core.manager
            },
            "javascript": {
                "linter": "eslint",
                "formatter": "prettier",
                "security": "npm audit"
            },
            "shell": {
                "linter": "shellcheck",
                "formatter": "shfmt"
            },
            "html": {
                "validator": "html-validator",
                "formatter": "prettier"
            }
        }

    def generate_code(self, description: str, language: str = "python", 
                     output_dir: str = None) -> Dict:
        """Generate code from natural language description."""
        try:
            # Parse description and identify requirements
            requirements = self._parse_description(description)
            
            # Select appropriate template
            template = self._select_template(language, requirements)
            
            # Generate code using AI
            code = self._generate_with_ai(description, language, template)
            
            # Format and validate code
            formatted_code = self._format_code(code, language)
            
            # Create output directory
            if output_dir:
                output_path = Path(output_dir)
            else:
                output_path = Path("generated_code") / f"{language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save generated code
            main_file = output_path / f"main.{self._get_file_extension(language)}"
            with open(main_file, "w") as f:
                f.write(formatted_code)
            
            # Generate additional files if needed
            self._generate_additional_files(output_path, language, requirements)
            
            # Run security audit
            if self.config["security"]["audit_enabled"]:
                security_issues = self._audit_code(main_file, language)
                self.generation_results["security_issues"].extend(security_issues)
            
            # Initialize git repository
            self._initialize_git_repo(output_path)
            
            self.generation_results["files_created"].append(str(main_file))
            return {
                "success": True,
                "output_dir": str(output_path),
                "main_file": str(main_file),
                "security_issues": self.generation_results["security_issues"]
            }
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            self.generation_results["errors"].append(str(e))
            return {
                "success": False,
                "error": str(e)
            }

    def _parse_description(self, description: str) -> Dict:
        """Parse natural language description to identify requirements."""
        # Use AI to parse description and extract requirements
        prompt = f"""
        Parse the following code generation request and extract requirements:
        {description}
        
        Return a JSON object with the following structure:
        {{
            "type": "web_app|api|script|library|exploit|network",
            "features": ["feature1", "feature2"],
            "dependencies": ["dep1", "dep2"],
            "security_requirements": ["req1", "req2"],
            "paradigm": "procedural|oop|functional|declarative",
            "style": "standard|custom",
            "testing": ["unit", "integration", "security"]
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.models["code_generation"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Failed to parse description: {e}")
            return {
                "type": "script",
                "features": [],
                "dependencies": [],
                "security_requirements": [],
                "paradigm": "procedural",
                "style": "standard",
                "testing": []
            }

    def _select_template(self, language: str, requirements: Dict) -> Any:
        """Select appropriate code template based on requirements."""
        if language not in self.templates:
            raise ValueError(f"Unsupported language: {language}")
            
        if requirements["type"] not in self.templates[language]:
            raise ValueError(f"Unsupported code type: {requirements['type']}")
            
        return self.templates[language][requirements["type"]]

    def _generate_with_ai(self, description: str, language: str, 
                         template: Any) -> str:
        """Generate code using AI model."""
        prompt = f"""
        Generate {language} code for the following task:
        {description}
        
        Use the following template structure:
        {template}
        
        Ensure the code:
        1. Follows best practices for {language}
        2. Includes proper error handling
        3. Has comprehensive comments
        4. Is secure and follows security best practices
        5. Is optimized for performance
        6. Follows the specified programming paradigm
        7. Includes proper documentation
        8. Has appropriate logging
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.models["code_generation"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config["ai"]["temperature"],
                max_tokens=self.config["ai"]["max_tokens"]
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"AI code generation failed: {e}")
            raise

    def _format_code(self, code: str, language: str) -> str:
        """Format code according to language-specific style guide."""
        if language == "python":
            try:
                # Format with black
                formatted = black.format_str(code, mode=black.FileMode())
                
                # Fix remaining style issues with autopep8
                formatted = autopep8.fix_code(formatted)
                
                return formatted
            except Exception as e:
                self.logger.warning(f"Code formatting failed: {e}")
                return code
        elif language == "javascript":
            try:
                # Use prettier for JavaScript formatting
                with tempfile.NamedTemporaryFile(mode='w', suffix='.js') as tmp:
                    tmp.write(code)
                    tmp.flush()
                    subprocess.run(["prettier", "--write", tmp.name])
                    with open(tmp.name, 'r') as f:
                        return f.read()
            except Exception as e:
                self.logger.warning(f"JavaScript formatting failed: {e}")
                return code
        else:
            return code

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for given language."""
        extensions = {
            "python": "py",
            "javascript": "js",
            "ruby": "rb",
            "c": "c",
            "go": "go",
            "shell": "sh",
            "html": "html"
        }
        return extensions.get(language, "txt")

    def _generate_additional_files(self, output_path: Path, 
                                 language: str, requirements: Dict):
        """Generate additional files like requirements.txt, README.md, etc."""
        # Generate requirements.txt for Python
        if language == "python" and requirements["dependencies"]:
            with open(output_path / "requirements.txt", "w") as f:
                f.write("\n".join(requirements["dependencies"]))
        
        # Generate README.md
        readme_content = f"""
        # Generated {language.capitalize()} Code
        
        ## Description
        This code was automatically generated based on the following requirements:
        {json.dumps(requirements, indent=2)}
        
        ## Setup
        1. Install dependencies: `pip install -r requirements.txt`
        2. Run the code: `python main.py`
        
        ## Security Considerations
        {", ".join(requirements["security_requirements"])}
        
        ## Testing
        Run tests: `python -m pytest`
        """
        
        with open(output_path / "README.md", "w") as f:
            f.write(readme_content)
        
        # Generate test files if testing is enabled
        if self.config["testing"]["auto_test"]:
            self._generate_test_files(output_path, language, requirements)

    def _generate_test_files(self, output_path: Path, 
                           language: str, requirements: Dict):
        """Generate test files for the code."""
        if language == "python":
            test_dir = output_path / "tests"
            test_dir.mkdir(exist_ok=True)
            
            # Generate test file
            test_content = f"""
            import pytest
            from main import *
            
            def test_main_function():
                # Add test cases here
                pass
            """
            
            with open(test_dir / "test_main.py", "w") as f:
                f.write(test_content)

    def _audit_code(self, file_path: Path, language: str) -> List[Dict]:
        """Run security audit on generated code."""
        issues = []
        
        if language == "python":
            try:
                # Run bandit for security analysis
                manager = bandit.core.manager.BanditManager()
                manager.discover_files([str(file_path)], None)
                manager.run_tests()
                
                for result in manager.results:
                    issues.append({
                        "type": "security",
                        "severity": result.severity,
                        "message": result.text,
                        "line": result.lineno
                    })
                
                # Run safety check for dependencies
                with tempfile.NamedTemporaryFile() as tmp:
                    subprocess.run(["pip", "freeze"], stdout=tmp)
                    safety_results = safety.cli.check(tmp.name)
                    
                    for result in safety_results:
                        issues.append({
                            "type": "dependency",
                            "severity": "high",
                            "message": f"Vulnerable dependency: {result.name} {result.version}",
                            "details": result.description
                        })
                
            except Exception as e:
                self.logger.error(f"Security audit failed: {e}")
        
        return issues

    def _initialize_git_repo(self, path: Path):
        """Initialize git repository for generated code."""
        try:
            repo = git.Repo.init(path)
            
            # Create .gitignore
            gitignore_content = """
            # Python
            __pycache__/
            *.py[cod]
            *$py.class
            .env
            venv/
            .venv/
            
            # Node.js
            node_modules/
            npm-debug.log*
            yarn-debug.log*
            yarn-error.log*
            
            # IDE
            .idea/
            .vscode/
            *.swp
            *.swo
            """
            
            with open(path / ".gitignore", "w") as f:
                f.write(gitignore_content)
            
            # Add and commit files
            repo.git.add(all=True)
            repo.git.commit(message="Initial commit: Generated code")
            
        except Exception as e:
            self.logger.error(f"Git initialization failed: {e}")

    def modify_code(self, file_path: str, modification_request: str) -> Dict:
        """Modify existing code based on natural language request."""
        try:
            with open(file_path, "r") as f:
                original_code = f.read()
            
            # Generate modification using AI
            prompt = f"""
            Modify the following code based on this request:
            {modification_request}
            
            Original code:
            {original_code}
            
            Return only the modified code, maintaining the same structure and style.
            """
            
            response = openai.ChatCompletion.create(
                model=self.models["code_generation"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            modified_code = response.choices[0].message.content
            
            # Write modified code
            with open(file_path, "w") as f:
                f.write(modified_code)
            
            return {
                "success": True,
                "file": file_path,
                "changes": self._diff_code(original_code, modified_code)
            }
            
        except Exception as e:
            self.logger.error(f"Code modification failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _diff_code(self, original: str, modified: str) -> List[str]:
        """Generate diff between original and modified code."""
        # Simple line-by-line diff
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()
        
        diff = []
        for i, (orig, mod) in enumerate(zip(original_lines, modified_lines)):
            if orig != mod:
                diff.append(f"Line {i+1}:")
                diff.append(f"- {orig}")
                diff.append(f"+ {mod}")
        
        return diff

    def run_tests(self, file_path: str) -> Dict:
        """Run tests on the generated code."""
        try:
            # Determine language from file extension
            language = self._get_language_from_extension(file_path)
            
            if language == "python":
                # Run pytest
                result = subprocess.run(
                    ["pytest", file_path],
                    capture_output=True,
                    text=True
                )
                
                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr
                }
            else:
                return {
                    "success": False,
                    "error": f"Testing not supported for {language}"
                }
                
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _get_language_from_extension(self, file_path: str) -> str:
        """Get programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".rb": "ruby",
            ".c": "c",
            ".go": "go",
            ".sh": "shell",
            ".html": "html"
        }
        return language_map.get(ext, "unknown")

def main():
    """Main function to run the auto coder tool."""
    try:
        auto_coder = AutoCoder()
        
        # Example usage
        description = """
        Create a Python script that:
        1. Takes a URL as input
        2. Checks for common web vulnerabilities
        3. Generates a report of findings
        4. Includes proper error handling and logging
        """
        
        result = auto_coder.generate_code(description, "python")
        
        if result["success"]:
            print(f"Code generated successfully in {result['output_dir']}")
            if result["security_issues"]:
                print("\nSecurity issues found:")
                for issue in result["security_issues"]:
                    print(f"- {issue['message']}")
        else:
            print(f"Error: {result['error']}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 