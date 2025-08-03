#!/usr/bin/env python3
"""
Documentation synchronization script for Social-xLSTM project.
Ensures architecture documentation stays in sync with code interfaces.

This script was created as part of the multi-LLM collaborative architecture design process:
- DeepSeek R1: Designed the documentation synchronization and validation system
- OpenAI o3-pro: Contributed interface validation and schema checking requirements
- Claude Opus 4: Provided architecture documentation structure requirements

Purpose: Maintain consistency between code interfaces and architecture documentation
Key Functions:
1. Validate interface documentation completeness against actual code
2. Check code examples in documentation for syntax correctness
3. Verify configuration schema validity through import testing
4. Generate interface documentation from code docstrings

Integration: Used in pre-commit hooks and CI/CD pipeline for documentation quality assurance

Usage: 
  python scripts/doc_sync.py --validate    # Check documentation sync
  python scripts/doc_sync.py --generate    # Generate interface docs from code
"""

import ast
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
import importlib.util


class DocSyncValidator:
    """
    Validates documentation sync with code.
    
    Design Pattern: Validator pattern for ensuring documentation-code consistency
    Core Responsibilities:
    - Interface Documentation Validation: Ensures all code interfaces are documented
    - Code Example Validation: Verifies syntax correctness of documentation examples
    - Schema Validation: Tests configuration classes for import/instantiation success
    - Documentation Generation: Creates interface docs from code docstrings
    
    Validation Strategy:
    1. AST Analysis: Parse Python files to extract interface definitions
    2. Documentation Cross-Reference: Check architecture docs contain all interfaces
    3. Code Example Testing: Parse and validate embedded Python code blocks
    4. Import Testing: Verify configuration schemas are importable and instantiable
    
    Integration Context:
    Part of the CI/CD pipeline ensuring documentation remains accurate as code evolves.
    Prevents documentation drift that could mislead developers implementing the architecture.
    """
    
    def __init__(self):
        """
        Initialize documentation synchronization validator.
        
        State Variables:
        - violations: Accumulates all synchronization violations found during validation
        """
        self.violations = []  # List of documentation sync violations
    
    def validate_interface_docs(self) -> bool:
        """
        Validate that interface documentation matches code.
        
        Returns:
            True if all interfaces are documented and examples are valid, False otherwise
        
        Validation Process:
        1. Locate interface code directory and architecture documentation
        2. Extract all interface class names from Python files
        3. Cross-reference interfaces against architecture documentation content
        4. Validate syntax of embedded code examples
        5. Report any missing documentation or invalid examples
        
        Interface Detection:
        - Classes ending in 'Interface' (naming convention)
        - Classes inheriting from Protocol (typing.Protocol)
        
        Documentation Requirements:
        - All interfaces must be mentioned in architecture documentation
        - Code examples in documentation must be syntactically valid
        - Architecture document must exist and be readable
        
        Complexity: O(N*M) where N=interface_files, M=interfaces_per_file
        """
        interfaces_path = Path('src/social_xlstm/interfaces')
        arch_doc_path = Path('docs/architecture/social_pooling.md')
        
        # Ensure architecture documentation exists
        if not arch_doc_path.exists():
            self.violations.append("Architecture document not found")
            return False
        
        # Read architecture document for interface cross-referencing
        with open(arch_doc_path, 'r', encoding='utf-8') as f:
            arch_content = f.read()
        
        # Check that all interface classes are documented
        for py_file in interfaces_path.glob('*.py'):
            if py_file.name.startswith('__'):
                continue  # Skip __init__.py and __pycache__
                
            # Extract interface class names from each Python file
            interfaces = self._extract_interfaces(py_file)
            for interface_name in interfaces:
                if interface_name not in arch_content:
                    self.violations.append(
                        f"Interface {interface_name} from {py_file} not documented"
                    )
        
        # Validate code examples in documentation for syntax correctness
        self._validate_code_examples(arch_content)
        
        return len(self.violations) == 0
    
    def validate_config_schemas(self) -> bool:
        """
        Validate that config schemas are up to date.
        
        Returns:
            True if all configuration classes can be imported and instantiated, False otherwise
        
        Validation Strategy:
        1. Dynamic Import: Load configuration module using importlib
        2. Class Instantiation: Test default construction of all config classes
        3. Exception Handling: Capture import errors, missing classes, and validation failures
        
        Configuration Classes Tested:
        - ModelConfig: Main model configuration with validation rules
        - SocialPoolingConfig: Social pooling parameters and constraints
        - XLSTMConfig: xLSTM-specific configuration settings
        
        Why This Matters:
        Configuration schemas use Pydantic for validation. Import/instantiation testing
        ensures schema definitions remain valid and don't break with code changes.
        
        Error Detection:
        - Import failures: Module syntax errors, missing dependencies
        - Class missing: Configuration classes renamed or removed
        - Validation errors: Default values violating Pydantic constraints
        """
        try:
            # Try to import configuration module using dynamic loading
            spec = importlib.util.spec_from_file_location(
                "config", "src/social_xlstm/interfaces/config.py"
            )
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Test basic instantiation of all configuration classes
            # This validates default values and Pydantic schema correctness
            config_module.ModelConfig()
            config_module.SocialPoolingConfig()
            config_module.XLSTMConfig()
            
        except Exception as e:
            self.violations.append(f"Config validation failed: {e}")
            return False
        
        return True
    
    def generate_interface_docs(self) -> str:
        """
        Generate documentation from interface docstrings.
        
        Returns:
            Markdown-formatted documentation string extracted from interface code
        
        Generation Process:
        1. Scan all Python files in interfaces directory
        2. Extract docstrings from classes and functions using AST parsing
        3. Format docstrings into structured Markdown documentation
        4. Organize by file with hierarchical section headers
        
        Documentation Structure:
        - File level: ## filename (H2 headers)
        - Class/Function level: ### name (H3 headers)
        - Content: Raw docstring content preserved
        
        Use Cases:
        - Automated documentation generation from code
        - Keeping interface docs in sync with code comments
        - CI/CD integration for documentation updates
        
        Output Format: Markdown string ready for writing to .md files
        """
        interfaces_path = Path('src/social_xlstm/interfaces')
        docs = []
        
        # Process each Python file in interfaces directory
        for py_file in interfaces_path.glob('*.py'):
            if py_file.name.startswith('__'):
                continue  # Skip Python package files
                
            # Add file-level header
            docs.append(f"## {py_file.stem}")
            docs.append("")
            
            # Extract and format docstrings from classes and functions
            docstrings = self._extract_docstrings(py_file)
            for name, docstring in docstrings.items():
                docs.append(f"### {name}")
                docs.append("")
                docs.append(docstring)
                docs.append("")
        
        return "\\n".join(docs)
    
    def _extract_interfaces(self, py_file: Path) -> List[str]:
        """
        Extract interface class names from Python file.
        
        Args:
            py_file: Path to Python source file to analyze
            
        Returns:
            List of interface class names found in the file
        
        Interface Detection Criteria:
        1. Naming Convention: Class name ends with 'Interface'
        2. Protocol Inheritance: Class inherits from typing.Protocol
        
        Implementation:
        - Uses AST parsing to analyze class definitions
        - Walks entire AST to find all class nodes
        - Applies interface detection heuristics to each class
        
        Error Handling:
        - Syntax errors in Python files are logged as warnings
        - Parse failures don't stop processing of other files
        - Returns empty list if file cannot be parsed
        
        Used By: validate_interface_docs() for documentation cross-referencing
        """
        interfaces = []
        
        try:
            # Parse Python file into AST for class analysis
            with open(py_file, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            # Walk AST to find all class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's an interface using naming convention or Protocol inheritance
                    if (node.name.endswith('Interface') or 
                        any(isinstance(base, ast.Name) and base.id == 'Protocol' 
                            for base in node.bases)):
                        interfaces.append(node.name)
        
        except Exception as e:
            print(f"Warning: Could not parse {py_file}: {e}")
        
        return interfaces
    
    def _extract_docstrings(self, py_file: Path) -> Dict[str, str]:
        """
        Extract docstrings from Python file.
        
        Args:
            py_file: Path to Python source file to analyze
            
        Returns:
            Dictionary mapping class/function names to their docstrings
        
        Extraction Process:
        1. Parse Python file into AST
        2. Walk AST to find class and function definitions
        3. Extract docstrings using ast.get_docstring()
        4. Map entity names to their documentation content
        
        Supported Entities:
        - Classes (ast.ClassDef): Class-level documentation
        - Functions (ast.FunctionDef): Function/method documentation
        - Module-level docstrings are not extracted (handled separately)
        
        Error Handling:
        - Syntax errors are logged as warnings but don't stop processing
        - Files without docstrings return empty dictionary
        - Missing docstrings for individual entities are skipped
        
        Used By: generate_interface_docs() for automated documentation generation
        """
        docstrings = {}
        
        try:
            # Parse Python file into AST for docstring extraction
            with open(py_file, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            # Walk AST to find classes and functions with docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        docstrings[node.name] = docstring
        
        except Exception as e:
            print(f"Warning: Could not extract docstrings from {py_file}: {e}")
        
        return docstrings
    
    def _validate_code_examples(self, content: str):
        """
        Validate code examples in documentation.
        
        Args:
            content: Markdown documentation content to validate
        
        Validation Process:
        1. Extract Python code blocks using regex pattern matching
        2. Parse each code block using Python AST parser
        3. Report syntax errors with code block location information
        4. Add violations to global violations list for reporting
        
        Code Block Detection:
        - Searches for ```python\n...\n``` markdown code blocks
        - Uses DOTALL regex flag to match multiline code examples
        - Handles nested code structures and complex examples
        
        Syntax Validation:
        - Uses ast.parse() for comprehensive Python syntax checking
        - Detects syntax errors, indentation issues, and malformed code
        - Provides specific error messages with code block numbers
        
        Error Reporting:
        - Violations include code block index for easy location
        - Syntax error details help developers fix documentation issues
        - Non-blocking: continues validation even if some blocks fail
        
        Used By: validate_interface_docs() for documentation quality assurance
        """
        import re
        
        # Find Python code blocks in Markdown documentation
        code_blocks = re.findall(r'```python\\n(.*?)\\n```', content, re.DOTALL)
        
        # Validate syntax of each code block
        for i, code_block in enumerate(code_blocks):
            try:
                # Basic syntax validation using Python AST parser
                ast.parse(code_block)
            except SyntaxError as e:
                self.violations.append(
                    f"Syntax error in code example {i+1}: {e}"
                )


def main():
    """
    Main function for documentation synchronization validation.
    
    Command-Line Interface:
    - --validate: Run comprehensive documentation sync validation
    - --generate: Generate interface documentation from code docstrings
    
    Validation Mode (--validate):
    1. Interface Documentation: Check all interfaces are documented in architecture docs
    2. Configuration Schemas: Verify config classes can be imported and instantiated
    3. Code Examples: Validate syntax of Python code blocks in documentation
    4. Comprehensive Reporting: List all violations with actionable details
    
    Generation Mode (--generate):
    - Extract docstrings from interface code files
    - Format as structured Markdown documentation
    - Output to stdout for redirection to documentation files
    
    Exit Codes:
    - 0: Success (validation passed or generation completed)
    - 1: Validation failures found
    
    Integration:
    - Pre-commit hooks: Run --validate to prevent documentation drift
    - CI/CD pipeline: Automated validation and documentation updates
    - Development workflow: Generate docs from code during development
    
    Error Handling:
    - Violations are collected and reported comprehensively
    - Individual failures don't stop overall validation process
    - Clear error messages guide developers to fix issues
    """
    parser = argparse.ArgumentParser(description='Documentation sync validator')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate documentation sync')
    parser.add_argument('--generate', action='store_true',
                       help='Generate interface documentation')
    
    args = parser.parse_args()
    
    # Initialize validator with violation tracking
    validator = DocSyncValidator()
    
    if args.generate:
        # Generate interface documentation from code docstrings
        docs = validator.generate_interface_docs()
        print(docs)
        return
    
    if args.validate:
        # Run all validations comprehensively
        validator.validate_interface_docs()
        validator.validate_config_schemas()
        
        # Report all violations or success
        if validator.violations:
            print("❌ Documentation sync violations:")
            for violation in validator.violations:
                print(f"  • {violation}")
            sys.exit(1)
        else:
            print("✅ Documentation is in sync with code")
    else:
        print("Use --validate or --generate")


if __name__ == '__main__':
    main()