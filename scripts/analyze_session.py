import argparse
import sqlite3
import json
import os
import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def analyze_session(db_path: str, output_file: str):
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        sys.exit(1)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        sys.exit(1)

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_runs'")
        if not cursor.fetchone():
            print("Error: Table 'agent_runs' not found in the database.")
            return

        cursor.execute("SELECT stage_name, role, success, metadata, output, created_at FROM agent_runs ORDER BY created_at ASC")
        rows = cursor.fetchall()
        
        if RICH_AVAILABLE:
            console = Console()
            console.print(f"\n[bold green]Session Analysis[/bold green] - [blue]{db_path}[/blue]")
            console.print(f"Total steps: [bold]{len(rows)}[/bold]\n")

            table = Table(title="Agent Runs Summary", box=box.ROUNDED)
            table.add_column("Step", justify="right", style="cyan", no_wrap=True)
            table.add_column("Stage", style="magenta")
            table.add_column("Role", style="blue")
            table.add_column("Status", justify="center")
            table.add_column("Details", style="dim")
        else:
            print(f"\nSession Analysis - {db_path}")
            print(f"Total steps: {len(rows)}\n")
            print(f"{'Step':>4} | {'Stage':<15} | {'Role':<15} | {'Status':<10} | {'Details'}")
            print("-" * 70)

        with open(output_file, "w", encoding="utf-8") as f:
            for i, (stage, role, success, metadata, output, created_at) in enumerate(rows):
                step_num = i + 1
                
                # Status string formatting
                status_text = "Success" if success == 1 else "Failed" if success == 0 else str(success)
                if RICH_AVAILABLE:
                    if success == 1:
                        status_ui = "[green]✓ Success[/green]"
                    elif success == 0:
                        status_ui = "[red]✗ Failed[/red]"
                    else:
                        status_ui = f"[yellow]? {success}[/yellow]"
                else:
                    status_ui = status_text

                details = []
                try:
                    if metadata:
                        meta = json.loads(metadata)
                        if isinstance(meta, dict):
                            tool_calls = meta.get("tool_calls", [])
                            if tool_calls:
                                failed_tool_calls = sum(1 for call in tool_calls if not call.get("success", False))
                                tool_label = f"{len(tool_calls)} tool calls"
                                if failed_tool_calls:
                                    tool_label += f" ({failed_tool_calls} failed)"
                                details.append(tool_label)
                                last_tool = tool_calls[-1].get("requested_tool_name") or tool_calls[-1].get("resolved_tool_name")
                                if last_tool:
                                    details.append(f"Last tool: {last_tool}")
                            spawn_reason = meta.get("spawn_reason")
                            if spawn_reason:
                                details.append(f"Spawn: {spawn_reason}")
                            if "error" in meta:
                                details.append(f"Error: {meta['error']}")
                    else:
                        details.append("No metadata")
                except Exception:
                    details.append("Invalid JSON metadata")

                details_str = ", ".join(details) if details else "None"

                if RICH_AVAILABLE:
                    table.add_row(str(step_num), str(stage), str(role), status_ui, details_str)
                else:
                    print(f"{step_num:>4} | {str(stage):<15} | {str(role):<15} | {status_ui:<10} | {details_str}")

                f.write(f"=== Step {step_num} [{stage} | {role}] ===\n")
                f.write(f"Status: {status_text}\n")
                f.write(f"Created At: {created_at}\n\n")
                f.write("--- Output ---\n")
                if output:
                    f.write(output[:1000] + ("\n... (truncated)" if len(output) > 1000 else "") + "\n")
                else:
                    f.write("None\n")
                f.write("\n")
                
                if metadata:
                    try:
                        f.write("--- Metadata ---\n")
                        f.write(json.dumps(json.loads(metadata), indent=2) + "\n")
                    except:
                        f.write(f"{metadata}\n")
                f.write("\n" + "="*50 + "\n\n")

        if RICH_AVAILABLE:
            console.print(table)
            console.print(f"\n[green]Detailed step summaries and outputs written to[/green] [bold]{output_file}[/bold]\n")
        else:
            print(f"\nDetailed step summaries and outputs written to {output_file}\n")

    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a mAIke session database.")
    parser.add_argument("db_path", help="Path to the session.db file")
    parser.add_argument("-o", "--output", default="/tmp/agent_runs_summary.txt", 
                        help="Path to write the detailed summary text file (default: /tmp/agent_runs_summary.txt)")
    
    args = parser.parse_args()
    analyze_session(args.db_path, args.output)
