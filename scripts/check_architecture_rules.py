#!/usr/bin/env python3
"""
Architecture compliance checker for Social-xLSTM project.
Validates that implementation follows architecture design principles.

This script was created as part of the multi-LLM collaborative architecture design process:
- DeepSeek R1: Designed the AST-based compliance checking system
- OpenAI o3-pro: Contributed interface validation requirements  
- Claude Opus 4: Provided distributed architecture design principles

Purpose: Enforce distributed Social-xLSTM architecture compliance through automated checks
Key Rules:
1. Use nn.ModuleDict instead of Python dict for neural network modules
2. Interface classes must inherit from Protocol for proper type checking
3. Prevent usage of deprecated centralized model implementations
4. Ensure proper parameter registration for PyTorch model components

Usage: python scripts/check_architecture_rules.py
Exit Code: 0 if compliant, 1 if violations found
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Any


class ArchitectureChecker(ast.NodeVisitor):
    """
    AST visitor to check architecture compliance.
    
    Implementation Pattern: Visitor pattern for traversing Python AST nodes
    Design Principles:
    - Stateful traversal: Tracks context (current file, model class state) 
    - Rule-based validation: Each visit method enforces specific architecture rules
    - Violation collection: Accumulates all violations for comprehensive reporting
    
    Compliance Rules Enforced:
    1. ModuleDict Usage: Neural network modules must use nn.ModuleDict for parameter registration
    2. Protocol Inheritance: Interface classes must inherit from Protocol
    3. Deprecated Model Prevention: Blocks usage of centralized model implementations
    4. Parameter Registration: Ensures proper PyTorch parameter handling
    
    Complexity: O(N) where N = number of AST nodes in source file
    """
    
    def __init__(self):
        """
        Initialize architecture compliance checker.
        
        State Variables:
        - violations: List of compliance violations found during traversal
        - current_file: Path object for currently analyzed file (for violation reporting)
        - in_model_class: Boolean flag tracking if currently inside PyTorch model class
        """
        self.violations = []          # Accumulated compliance violations
        self.current_file = None      # Current file being analyzed
        self.in_model_class = False   # Context flag for PyTorch model class detection
    
    def check_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """
        Check a single Python file for architecture violations.
        
        Args:
            filepath: Path to Python source file to analyze
            
        Returns:
            List of violation dictionaries containing:
            - rule: Violated rule identifier (e.g., 'module-dict-required')
            - message: Human-readable violation description
            - file: Path to file containing violation
            - line: Line number where violation occurs
            - fix_hint: Optional suggestion for fixing the violation
        
        Processing Flow:
        1. Parse file into Abstract Syntax Tree (AST)
        2. Traverse AST using visitor pattern
        3. Apply architecture compliance rules at each node
        4. Collect violations with location and fix hints
        
        Error Handling:
        - Syntax errors are captured as violations rather than exceptions
        - Missing files or permission errors propagate to caller
        """
        self.current_file = filepath
        self.violations = []  # Reset violations for this file
        
        try:
            # Parse Python source into AST for rule-based analysis
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(filepath))
            
            # Traverse AST applying architecture compliance rules
            self.visit(tree)
            
        except SyntaxError as e:
            # Treat syntax errors as architecture violations for consistent reporting
            self.violations.append({
                'rule': 'syntax-error',
                'message': f'Syntax error: {e}',
                'file': filepath,
                'line': getattr(e, 'lineno', 0)
            })
        
        return self.violations
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """
        Check class definitions for architecture compliance.
        
        Architecture Rules Enforced:
        1. PyTorch Model Detection: Identifies classes inheriting from Module/LightningModule
        2. Interface Protocol Requirement: Interface classes must inherit from Protocol
        
        Args:
            node: AST ClassDef node representing a class definition
        
        State Changes:
        - Sets in_model_class flag for downstream ModuleDict validation
        - Resets in_model_class flag after processing class body
        
        Violations Detected:
        - interface-protocol-required: Interface classes without Protocol inheritance
        """
        # Check if it's a PyTorch model class for ModuleDict validation context
        if any(base.id in ['Module', 'LightningModule'] for base in node.bases 
               if isinstance(base, ast.Name)):
            self.in_model_class = True
            
        # Architecture Rule: Interface classes must inherit from Protocol for proper typing
        if node.name.endswith('Interface') and not self._has_protocol_base(node):
            self.violations.append({
                'rule': 'interface-protocol-required',
                'message': f'Interface class {node.name} should inherit from Protocol',
                'file': self.current_file,
                'line': node.lineno
            })
            
        # Continue AST traversal for nested classes and methods
        self.generic_visit(node)
        
        # Reset model class context after processing class body
        self.in_model_class = False
    
    def visit_Assign(self, node: ast.Assign):
        """
        Check assignments for ModuleDict usage in model classes.
        
        Architecture Rule: PyTorch model classes must use nn.ModuleDict instead of Python dict
        for neural network module storage to ensure proper parameter registration.
        
        Args:
            node: AST Assign node representing an assignment statement
        
        Detection Logic:
        1. Only check assignments within PyTorch model classes (in_model_class=True)
        2. Identify dict assignments that should be ModuleDict based on heuristics
        3. Use _looks_like_module_dict() to detect neural network module patterns
        
        Violations Detected:
        - module-dict-required: Python dict used instead of nn.ModuleDict for NN modules
        
        Why This Matters:
        nn.ModuleDict enables automatic parameter registration, gradient computation,
        and device placement for PyTorch models - critical for distributed training.
        """
        if self.in_model_class and isinstance(node.value, ast.Dict):
            # Check if this dict assignment should be a ModuleDict based on content heuristics
            if self._looks_like_module_dict(node):
                self.violations.append({
                    'rule': 'module-dict-required',
                    'message': 'Use nn.ModuleDict instead of Python dict for neural network modules',
                    'file': self.current_file,
                    'line': node.lineno,
                    'fix_hint': 'Replace dict with nn.ModuleDict for proper parameter registration'
                })
        
        # Continue AST traversal for nested assignments
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """
        Check function calls for architecture compliance.
        
        Architecture Rule: Prevent usage of deprecated centralized model implementations
        that violate the distributed Social-xLSTM design principles.
        
        Args:
            node: AST Call node representing a function/constructor call
        
        Detection Logic:
        - Identify calls to deprecated centralized model classes
        - Flag usage of legacy implementations that should be migrated
        
        Violations Detected:
        - deprecated-centralized-model: Usage of centralized implementations
        
        Migration Context:
        Part of the Phase 0 cleanup identifying code that needs migration to
        distributed architecture based on ADR-0100 decision.
        """
        # Check for deprecated centralized model usage that violates distributed architecture
        if (isinstance(node.func, ast.Name) and 
            node.func.id in ['SocialTrafficModel', 'CentralizedSocialXLSTM']):
            self.violations.append({
                'rule': 'deprecated-centralized-model',
                'message': f'Use of deprecated centralized model: {node.func.id}',
                'file': self.current_file,
                'line': node.lineno,
                'fix_hint': 'Use DistributedSocialXLSTMModel instead'
            })
        
        # Continue AST traversal for nested function calls
        self.generic_visit(node)
    
    def _has_protocol_base(self, node: ast.ClassDef) -> bool:
        """
        Check if class inherits from Protocol.
        
        Args:
            node: AST ClassDef node to check for Protocol inheritance
            
        Returns:
            True if class inherits from Protocol, False otherwise
        
        Implementation:
        Examines the base class list for explicit Protocol inheritance,
        which is required for proper interface typing in Social-xLSTM.
        
        Used By: visit_ClassDef() for interface-protocol-required rule validation
        """
        return any(isinstance(base, ast.Name) and base.id == 'Protocol' 
                  for base in node.bases)
    
    def _looks_like_module_dict(self, node: ast.Assign) -> bool:
        """
        Heuristic to detect if dict should be ModuleDict.
        
        Args:
            node: AST Assign node representing a dictionary assignment
            
        Returns:
            True if assignment appears to be a neural network module dictionary
            
        Detection Heuristics:
        1. Assignment Target: Must be assigned to self.attribute (instance variable)
        2. Dict Content Analysis: Dict values contain neural network module constructors
        3. Module Keywords: Looks for 'lstm', 'xlstm', 'linear', 'conv', 'module' in names
        
        Why This Matters:
        PyTorch requires nn.ModuleDict for proper parameter registration. Python dict
        won't register parameters correctly, breaking gradient computation and device placement.
        
        Examples Detected:
        - self.xlstm_modules = {'vd1': XLSTM(), 'vd2': XLSTM()}  # Should be ModuleDict
        - self.layers = {'linear': nn.Linear(), 'conv': nn.Conv1d()}  # Should be ModuleDict
        """
        # Check if assigned to self.something (instance variable pattern)
        if (len(node.targets) == 1 and 
            isinstance(node.targets[0], ast.Attribute) and
            isinstance(node.targets[0].value, ast.Name) and
            node.targets[0].value.id == 'self'):
            
            # Check if dict values look like neural network modules
            if isinstance(node.value, ast.Dict):
                for value in node.value.values:
                    if (isinstance(value, ast.Call) and 
                        isinstance(value.func, ast.Name) and
                        any(term in value.func.id.lower() 
                            for term in ['lstm', 'xlstm', 'linear', 'conv', 'module'])):
                        return True
        return False


def main():
    """
    Main function to run architecture checks.
    
    Workflow:
    1. Locate Social-xLSTM source directory (src/social_xlstm)
    2. Initialize ArchitectureChecker with compliance rules
    3. Recursively scan all Python files in source tree
    4. Apply architecture compliance rules to each file
    5. Report violations with actionable fix hints
    6. Exit with appropriate code (0=success, 1=violations)
    
    Exit Codes:
    - 0: All architecture rules passed
    - 1: Architecture violations found or source directory missing
    
    Output Format:
    For each violation:
    - 📁 File path and line number
    - 🚫 Rule name and violation message  
    - 💡 Fix hint (when available)
    
    Integration:
    Called by pre-commit hooks and CI/CD pipeline for continuous compliance validation
    """
    src_path = Path('src/social_xlstm')
    if not src_path.exists():
        print("Error: src/social_xlstm directory not found")
        sys.exit(1)
    
    # Initialize checker with distributed architecture compliance rules
    checker = ArchitectureChecker()
    all_violations = []
    
    # Check all Python files in the source directory recursively
    for py_file in src_path.rglob('*.py'):
        violations = checker.check_file(py_file)
        all_violations.extend(violations)
    
    # Report violations with user-friendly formatting
    if all_violations:
        print(f"❌ Found {len(all_violations)} architecture violations:")
        print()
        
        for violation in all_violations:
            print(f"📁 {violation['file']}:{violation['line']}")
            print(f"🚫 {violation['rule']}: {violation['message']}")
            if 'fix_hint' in violation:
                print(f"💡 Fix: {violation['fix_hint']}")
            print()
        
        # Exit with error code to fail CI/CD on violations
        sys.exit(1)
    else:
        print("✅ All architecture rules passed!")


if __name__ == '__main__':
    main()