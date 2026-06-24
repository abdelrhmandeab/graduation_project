"""
Phase 3 Implementation: Advanced Operations Handler

Handles command chaining, batch file operations, and semantic search.
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from core.logger import logger


@dataclass
class ChainedCommand:
    """Represents a single command in a chain."""
    text: str
    intent: str
    action: str
    args: Dict[str, Any]


class CommandChainManager:
    """Manage chained command execution."""
    
    def __init__(self):
        self.chain_commands: List[ChainedCommand] = []
        self.current_index = 0
        
    def parse_chain(self, command_text: str) -> List[ChainedCommand]:
        """Parse a chained command into individual commands."""
        # Split by conjunctions
        conjunction_pattern = re.compile(
            r'(?:\s+(?:and|or|then)\s+|\s+(?:و|او|أو|بعدين)\s+)',
            re.IGNORECASE
        )
        
        parts = conjunction_pattern.split(command_text)
        commands = []
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Each part would need to be parsed by main parser
            # For now, just store as-is
            commands.append(ChainedCommand(
                text=part,
                intent="",  # Will be populated by main parser
                action="",
                args={},
            ))
        
        return commands
    
    def execute_chain(self, commands: List[ChainedCommand], join_type: str = "sequence") -> Dict[str, Any]:
        """Execute chained commands in sequence or until success (OR)."""
        results = []
        
        for i, cmd in enumerate(commands):
            logger.info(f"Executing chain command {i+1}/{len(commands)}: {cmd.text}")
            # Parse and execute each command
            from core.command_parser import parse_command
            parsed = parse_command(cmd.text)
            
            results.append({
                "command": cmd.text,
                "intent": parsed.intent,
                "action": parsed.action,
                "args": parsed.args,
                "status": "parsed",  # Will be "executed" after handler processes it
            })
            
            # For OR join type, stop if command succeeded
            if join_type == "or" and parsed.intent != "LLM_QUERY":
                logger.info(f"Chain stopping at command {i+1} (OR mode, success)")
                break
        
        return {
            "join_type": join_type,
            "commands_executed": len(results),
            "results": results,
            "status": "ready_for_orchestrator",
        }


class BatchFileOperationManager:
    """Manage batch file operations."""
    
    def __init__(self):
        self.batch_queue: List[Dict[str, Any]] = []
        
    def parse_batch_delete(self, files_str: str, location: str = "") -> Dict[str, Any]:
        """Parse batch delete operation."""
        # Split files by comma or 'and' or 'و'
        file_list = re.split(r'(?:,\s*|(?:\s+and\s+)|(?:\s+و\s+))', files_str)
        file_list = [f.strip() for f in file_list if f.strip()]
        
        return {
            "operation": "delete_multiple",
            "files": file_list,
            "location": location.strip() if location else None,
            "count": len(file_list),
        }
    
    def parse_batch_copy(self, files_str: str, destination: str) -> Dict[str, Any]:
        """Parse batch copy operation."""
        file_list = re.split(r'(?:,\s*|(?:\s+and\s+)|(?:\s+و\s+))', files_str)
        file_list = [f.strip() for f in file_list if f.strip()]
        
        return {
            "operation": "copy_multiple",
            "files": file_list,
            "destination": destination.strip(),
            "count": len(file_list),
        }
    
    def parse_batch_move(self, files_str: str, destination: str) -> Dict[str, Any]:
        """Parse batch move operation."""
        file_list = re.split(r'(?:,\s*|(?:\s+and\s+)|(?:\s+و\s+))', files_str)
        file_list = [f.strip() for f in file_list if f.strip()]
        
        return {
            "operation": "move_multiple",
            "files": file_list,
            "destination": destination.strip(),
            "count": len(file_list),
        }


class SemanticSearchManager:
    """Manage advanced semantic file search."""
    
    def __init__(self):
        self.search_cache: Dict[str, List[str]] = {}
        
    def rank_by_recency(self, files: List[str]) -> List[str]:
        """Sort files by modification time (recent first)."""
        import os
        try:
            files_with_time = [(f, os.path.getmtime(f)) for f in files if os.path.exists(f)]
            files_with_time.sort(key=lambda x: x[1], reverse=True)
            return [f[0] for f in files_with_time]
        except Exception as e:
            logger.warning(f"Error ranking by recency: {e}")
            return files
    
    def rank_by_relevance(self, files: List[str], query: str) -> List[str]:
        """Sort files by relevance to search query."""
        # Simple relevance: exact match > partial match > other
        query_lower = query.lower()
        exact = []
        partial = []
        other = []
        
        for f in files:
            f_lower = f.lower()
            if query_lower in f_lower:
                if f_lower.startswith(query_lower) or f_lower.endswith(query_lower):
                    exact.append(f)
                else:
                    partial.append(f)
            else:
                other.append(f)
        
        return exact + partial + other
    
    def search(self, query: str, root: Optional[str] = None) -> Dict[str, Any]:
        """Perform semantic search on files."""
        import os
        
        if not root:
            root = os.path.expanduser("~")
        
        # Search for files matching query (basic implementation)
        results = []
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                # Limit depth to avoid infinite recursion
                depth = dirpath.count(os.sep) - root.count(os.sep)
                if depth > 5:
                    dirnames.clear()
                    continue
                
                for filename in filenames[:50]:  # Limit per directory
                    if query.lower() in filename.lower():
                        results.append(os.path.join(dirpath, filename))
                
                if len(results) > 20:  # Stop at 20 results
                    break
        except PermissionError:
            logger.warning(f"Permission denied searching: {root}")
        
        # Rank results: recency + relevance
        results = self.rank_by_recency(results[:10])
        results = self.rank_by_relevance(results, query)
        
        return {
            "query": query,
            "root": root,
            "results": results[:5],  # Top 5 results
            "count": len(results),
            "status": "complete",
        }


# Public API

chain_manager = CommandChainManager()
batch_manager = BatchFileOperationManager()
search_manager = SemanticSearchManager()


def handle_command_chain(command_text: str) -> Dict[str, Any]:
    """Handle a chained command."""
    logger.info(f"Handling command chain: {command_text}")
    commands = chain_manager.parse_chain(command_text)
    result = chain_manager.execute_chain(commands)
    return result


def handle_batch_file_operation(operation: str, **kwargs) -> Dict[str, Any]:
    """Handle a batch file operation."""
    logger.info(f"Handling batch file operation: {operation}")
    
    if operation == "delete_multiple":
        return batch_manager.parse_batch_delete(kwargs.get("files", ""), kwargs.get("location", ""))
    elif operation == "copy_multiple":
        return batch_manager.parse_batch_copy(kwargs.get("files", ""), kwargs.get("destination", ""))
    elif operation == "move_multiple":
        return batch_manager.parse_batch_move(kwargs.get("files", ""), kwargs.get("destination", ""))
    
    return {"error": f"Unknown batch operation: {operation}"}


def handle_semantic_search(query: str, root: Optional[str] = None) -> Dict[str, Any]:
    """Handle semantic file search."""
    logger.info(f"Handling semantic search: query='{query}', root='{root}'")
    return search_manager.search(query, root)
