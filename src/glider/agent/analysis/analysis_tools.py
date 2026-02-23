"""
Analysis Tools - CSV analysis functions for the AI agent.

Provides tools for loading, parsing, and analyzing tracking CSV data.
"""

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from glider.agent.llm_backend import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass
class LoadedFile:
    """Information about a loaded CSV file."""

    file_id: str
    file_path: str
    file_name: str
    data: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    row_count: int = 0
    duration_ms: float = 0.0
    columns: list[str] = field(default_factory=list)


# Tool definitions for the LLM
ANALYSIS_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="get_column_statistics",
        description="Calculate statistics (mean, std, min, max, median) for a numeric column. "
        "Use this to answer questions like 'what was the average velocity?' or 'mean distance traveled'.",
        parameters={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The ID of the loaded file to analyze",
                },
                "column": {
                    "type": "string",
                    "description": "Column name to analyze (e.g., 'velocity_px_frame', 'distance_mm', 'cumulative_mm')",
                },
                "start_ms": {
                    "type": "number",
                    "description": "Optional start time in milliseconds to filter data",
                },
                "end_ms": {
                    "type": "number",
                    "description": "Optional end time in milliseconds to filter data",
                },
            },
            "required": ["file_id", "column"],
        },
    ),
    ToolDefinition(
        name="analyze_zone_occupancy",
        description="Calculate time spent in each zone, number of entries/exits, and percentage of total time. "
        "Use this to answer questions about zone behavior.",
        parameters={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The ID of the loaded file to analyze",
                },
                "start_ms": {
                    "type": "number",
                    "description": "Optional start time in milliseconds",
                },
                "end_ms": {
                    "type": "number",
                    "description": "Optional end time in milliseconds",
                },
            },
            "required": ["file_id"],
        },
    ),
    ToolDefinition(
        name="analyze_behavioral_states",
        description="Analyze the distribution of behavioral states (FREEZE, IMMOBILE, MOVING, DARTING). "
        "Returns time in each state and percentage of total time.",
        parameters={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The ID of the loaded file to analyze",
                },
                "start_ms": {
                    "type": "number",
                    "description": "Optional start time in milliseconds",
                },
                "end_ms": {
                    "type": "number",
                    "description": "Optional end time in milliseconds",
                },
            },
            "required": ["file_id"],
        },
    ),
    ToolDefinition(
        name="calculate_distance_metrics",
        description="Calculate distance and movement metrics: total distance, average velocity, max velocity, "
        "time spent moving. Use for questions about movement and locomotion.",
        parameters={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The ID of the loaded file to analyze",
                },
                "start_ms": {
                    "type": "number",
                    "description": "Optional start time in milliseconds",
                },
                "end_ms": {
                    "type": "number",
                    "description": "Optional end time in milliseconds",
                },
            },
            "required": ["file_id"],
        },
    ),
    ToolDefinition(
        name="compare_time_periods",
        description="Compare metrics between two time periods. Useful for comparing first vs second half, "
        "or before vs after an event.",
        parameters={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The ID of the loaded file to analyze",
                },
                "period1_start_ms": {
                    "type": "number",
                    "description": "Start of first period in milliseconds",
                },
                "period1_end_ms": {
                    "type": "number",
                    "description": "End of first period in milliseconds",
                },
                "period2_start_ms": {
                    "type": "number",
                    "description": "Start of second period in milliseconds",
                },
                "period2_end_ms": {
                    "type": "number",
                    "description": "End of second period in milliseconds",
                },
            },
            "required": [
                "file_id",
                "period1_start_ms",
                "period1_end_ms",
                "period2_start_ms",
                "period2_end_ms",
            ],
        },
    ),
    ToolDefinition(
        name="get_data_sample",
        description="Get sample rows from the data for inspection. Useful for understanding the data structure.",
        parameters={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The ID of the loaded file",
                },
                "start_row": {
                    "type": "integer",
                    "description": "Starting row index (0-based)",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of rows to return (max 20)",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific columns to include",
                },
            },
            "required": ["file_id"],
        },
    ),
    ToolDefinition(
        name="list_loaded_files",
        description="List all currently loaded CSV files with their summaries.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    ToolDefinition(
        name="get_file_summary",
        description="Get a detailed summary of a loaded file including metadata, columns, and basic statistics.",
        parameters={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The ID of the loaded file",
                },
            },
            "required": ["file_id"],
        },
    ),
]


class AnalysisToolExecutor:
    """Executes analysis tools on loaded CSV data."""

    def __init__(self):
        """Initialize the tool executor."""
        self._loaded_files: dict[str, LoadedFile] = {}

    def clear(self) -> None:
        """Clear all loaded files."""
        self._loaded_files.clear()

    def get_loaded_files_summary(self) -> list[dict[str, Any]]:
        """Get summary of all loaded files."""
        return [
            {
                "file_id": f.file_id,
                "file_name": f.file_name,
                "row_count": f.row_count,
                "duration_seconds": f.duration_ms / 1000.0,
                "subject_id": f.metadata.get("subject_id", "Unknown"),
            }
            for f in self._loaded_files.values()
        ]

    def load_file(self, file_path: str) -> dict[str, Any]:
        """
        Load a CSV file and return summary.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary with file info and summary
        """
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        if not path.suffix.lower() == ".csv":
            return {"error": f"Not a CSV file: {file_path}"}

        try:
            # Parse metadata from header comments
            metadata = self._parse_metadata(path)

            # Read CSV, skipping comment lines
            df = pd.read_csv(path, comment="#")

            # Generate file ID
            file_id = str(uuid.uuid4())[:8]

            # Calculate duration
            duration_ms = 0.0
            if "elapsed_ms" in df.columns and len(df) > 0:
                duration_ms = df["elapsed_ms"].max()

            # Store loaded file
            loaded = LoadedFile(
                file_id=file_id,
                file_path=str(path),
                file_name=path.name,
                data=df,
                metadata=metadata,
                row_count=len(df),
                duration_ms=duration_ms,
                columns=list(df.columns),
            )
            self._loaded_files[file_id] = loaded

            logger.info(f"Loaded CSV file: {path.name} ({len(df)} rows)")

            return {
                "file_id": file_id,
                "file_name": path.name,
                "row_count": len(df),
                "columns": list(df.columns),
                "duration_seconds": duration_ms / 1000.0,
                "metadata": metadata,
            }

        except Exception as e:
            logger.exception(f"Failed to load CSV: {e}")
            return {"error": f"Failed to load CSV: {str(e)}"}

    def _parse_metadata(self, path: Path) -> dict[str, Any]:
        """Parse metadata from CSV header comments."""
        metadata = {}
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.startswith("#"):
                        break
                    # Parse comment lines like "# Subject ID,M001"
                    line = line[1:].strip()  # Remove # prefix
                    if "," in line:
                        parts = line.split(",", 1)
                        key = parts[0].strip().lower().replace(" ", "_")
                        value = parts[1].strip() if len(parts) > 1 else ""
                        metadata[key] = value
        except Exception as e:
            logger.warning(f"Failed to parse metadata: {e}")

        return metadata

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool result as dictionary
        """
        method = getattr(self, f"_tool_{tool_name}", None)
        if method is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return method(**arguments)
        except Exception as e:
            logger.exception(f"Tool execution failed: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}

    def _get_file(self, file_id: str) -> LoadedFile | None:
        """Get a loaded file by ID."""
        return self._loaded_files.get(file_id)

    def _filter_by_time(
        self,
        df: pd.DataFrame,
        start_ms: float | None = None,
        end_ms: float | None = None,
    ) -> pd.DataFrame:
        """Filter dataframe by time range."""
        if "elapsed_ms" not in df.columns:
            return df

        result = df
        if start_ms is not None:
            result = result[result["elapsed_ms"] >= start_ms]
        if end_ms is not None:
            result = result[result["elapsed_ms"] <= end_ms]
        return result

    # Tool implementations

    def _tool_list_loaded_files(self) -> dict[str, Any]:
        """List all loaded files."""
        if not self._loaded_files:
            return {"files": [], "message": "No files loaded"}

        return {"files": self.get_loaded_files_summary()}

    def _tool_get_file_summary(self, file_id: str) -> dict[str, Any]:
        """Get detailed summary of a file."""
        loaded = self._get_file(file_id)
        if not loaded:
            return {"error": f"File not found: {file_id}"}

        df = loaded.data

        # Get numeric columns for statistics
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        stats = {}
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            stats[col] = {
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
            }

        return {
            "file_id": file_id,
            "file_name": loaded.file_name,
            "row_count": loaded.row_count,
            "duration_seconds": loaded.duration_ms / 1000.0,
            "columns": loaded.columns,
            "metadata": loaded.metadata,
            "sample_statistics": stats,
        }

    def _tool_get_column_statistics(
        self,
        file_id: str,
        column: str,
        start_ms: float | None = None,
        end_ms: float | None = None,
    ) -> dict[str, Any]:
        """Calculate statistics for a column."""
        loaded = self._get_file(file_id)
        if not loaded:
            return {"error": f"File not found: {file_id}"}

        if column not in loaded.data.columns:
            return {"error": f"Column '{column}' not found. Available: {loaded.columns}"}

        df = self._filter_by_time(loaded.data, start_ms, end_ms)

        if len(df) == 0:
            return {"error": "No data in specified time range"}

        col_data = pd.to_numeric(df[column], errors="coerce")
        valid_data = col_data.dropna()

        if len(valid_data) == 0:
            return {"error": f"No valid numeric data in column '{column}'"}

        return {
            "column": column,
            "count": int(len(valid_data)),
            "mean": float(valid_data.mean()),
            "std": float(valid_data.std()),
            "min": float(valid_data.min()),
            "max": float(valid_data.max()),
            "median": float(valid_data.median()),
            "time_range_ms": {
                "start": start_ms or 0,
                "end": end_ms or loaded.duration_ms,
            },
        }

    def _tool_analyze_zone_occupancy(
        self,
        file_id: str,
        start_ms: float | None = None,
        end_ms: float | None = None,
    ) -> dict[str, Any]:
        """Analyze time spent in each zone."""
        loaded = self._get_file(file_id)
        if not loaded:
            return {"error": f"File not found: {file_id}"}

        if "zone_ids" not in loaded.data.columns:
            return {"error": "No zone_ids column in data"}

        df = self._filter_by_time(loaded.data, start_ms, end_ms)

        if len(df) == 0:
            return {"error": "No data in specified time range"}

        # Calculate frame duration (assuming consistent frame rate)
        if "elapsed_ms" in df.columns and len(df) > 1:
            frame_duration_ms = df["elapsed_ms"].diff().median()
        else:
            frame_duration_ms = 33.3  # Assume ~30fps

        total_duration_ms = len(df) * frame_duration_ms

        # Count frames per zone
        zone_counts: dict[str, int] = {}
        zone_entries: dict[str, int] = {}
        prev_zones: set[str] = set()

        for _, row in df.iterrows():
            zone_str = str(row.get("zone_ids", ""))
            current_zones = (
                {z.strip() for z in zone_str.split(",") if z.strip()} if zone_str else set()
            )

            for zone in current_zones:
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
                # Count entry if zone wasn't in previous frame
                if zone not in prev_zones:
                    zone_entries[zone] = zone_entries.get(zone, 0) + 1

            prev_zones = current_zones

        # Build results
        zones = {}
        for zone_name, count in zone_counts.items():
            time_ms = count * frame_duration_ms
            zones[zone_name] = {
                "time_ms": float(time_ms),
                "time_seconds": float(time_ms / 1000.0),
                "percentage": (
                    float(time_ms / total_duration_ms * 100) if total_duration_ms > 0 else 0
                ),
                "entries": zone_entries.get(zone_name, 0),
            }

        return {
            "zones": zones,
            "total_duration_seconds": float(total_duration_ms / 1000.0),
            "frame_count": len(df),
        }

    def _tool_analyze_behavioral_states(
        self,
        file_id: str,
        start_ms: float | None = None,
        end_ms: float | None = None,
    ) -> dict[str, Any]:
        """Analyze behavioral state distribution."""
        loaded = self._get_file(file_id)
        if not loaded:
            return {"error": f"File not found: {file_id}"}

        if "behavioral_state" not in loaded.data.columns:
            return {"error": "No behavioral_state column in data"}

        df = self._filter_by_time(loaded.data, start_ms, end_ms)

        if len(df) == 0:
            return {"error": "No data in specified time range"}

        # Calculate frame duration
        if "elapsed_ms" in df.columns and len(df) > 1:
            frame_duration_ms = df["elapsed_ms"].diff().median()
        else:
            frame_duration_ms = 33.3

        total_duration_ms = len(df) * frame_duration_ms

        # Count states
        state_counts = df["behavioral_state"].value_counts()

        states = {}
        for state, count in state_counts.items():
            if pd.isna(state) or state == "":
                state = "unknown"
            time_ms = count * frame_duration_ms
            states[str(state)] = {
                "frame_count": int(count),
                "time_ms": float(time_ms),
                "time_seconds": float(time_ms / 1000.0),
                "percentage": (
                    float(time_ms / total_duration_ms * 100) if total_duration_ms > 0 else 0
                ),
            }

        return {
            "states": states,
            "total_duration_seconds": float(total_duration_ms / 1000.0),
            "frame_count": len(df),
        }

    def _tool_calculate_distance_metrics(
        self,
        file_id: str,
        start_ms: float | None = None,
        end_ms: float | None = None,
    ) -> dict[str, Any]:
        """Calculate distance and movement metrics."""
        loaded = self._get_file(file_id)
        if not loaded:
            return {"error": f"File not found: {file_id}"}

        df = self._filter_by_time(loaded.data, start_ms, end_ms)

        if len(df) == 0:
            return {"error": "No data in specified time range"}

        # Use mm if available, otherwise px
        distance_col = "distance_mm" if "distance_mm" in df.columns else "distance_px"
        cumulative_col = "cumulative_mm" if "cumulative_mm" in df.columns else None
        velocity_col = "velocity_px_frame" if "velocity_px_frame" in df.columns else None

        result: dict[str, Any] = {"unit": "mm" if "mm" in distance_col else "px"}

        # Total distance
        if distance_col in df.columns:
            distances = pd.to_numeric(df[distance_col], errors="coerce").dropna()
            result["total_distance"] = float(distances.sum())
            result["average_distance_per_frame"] = float(distances.mean())

        # Cumulative distance (take max if available)
        if cumulative_col and cumulative_col in df.columns:
            cumulative = pd.to_numeric(df[cumulative_col], errors="coerce").dropna()
            if len(cumulative) > 0:
                result["final_cumulative_distance"] = float(cumulative.iloc[-1])

        # Velocity stats
        if velocity_col and velocity_col in df.columns:
            velocities = pd.to_numeric(df[velocity_col], errors="coerce").dropna()
            if len(velocities) > 0:
                result["velocity"] = {
                    "mean": float(velocities.mean()),
                    "max": float(velocities.max()),
                    "std": float(velocities.std()),
                }

        # Time moving (behavioral state = MOVING or DARTING)
        if "behavioral_state" in df.columns:
            moving_mask = df["behavioral_state"].isin(["MOVING", "DARTING"])
            moving_frames = moving_mask.sum()
            total_frames = len(df)
            result["time_moving"] = {
                "frames": int(moving_frames),
                "percentage": float(moving_frames / total_frames * 100) if total_frames > 0 else 0,
            }

        result["frame_count"] = len(df)

        return result

    def _tool_compare_time_periods(
        self,
        file_id: str,
        period1_start_ms: float,
        period1_end_ms: float,
        period2_start_ms: float,
        period2_end_ms: float,
    ) -> dict[str, Any]:
        """Compare metrics between two time periods."""
        loaded = self._get_file(file_id)
        if not loaded:
            return {"error": f"File not found: {file_id}"}

        # Get metrics for each period
        period1_distance = self._tool_calculate_distance_metrics(
            file_id, period1_start_ms, period1_end_ms
        )
        period2_distance = self._tool_calculate_distance_metrics(
            file_id, period2_start_ms, period2_end_ms
        )

        period1_behavior = self._tool_analyze_behavioral_states(
            file_id, period1_start_ms, period1_end_ms
        )
        period2_behavior = self._tool_analyze_behavioral_states(
            file_id, period2_start_ms, period2_end_ms
        )

        return {
            "period1": {
                "time_range_ms": {
                    "start": period1_start_ms,
                    "end": period1_end_ms,
                },
                "distance_metrics": period1_distance,
                "behavioral_states": period1_behavior.get("states", {}),
            },
            "period2": {
                "time_range_ms": {
                    "start": period2_start_ms,
                    "end": period2_end_ms,
                },
                "distance_metrics": period2_distance,
                "behavioral_states": period2_behavior.get("states", {}),
            },
        }

    def _tool_get_data_sample(
        self,
        file_id: str,
        start_row: int = 0,
        count: int = 10,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get sample rows from the data."""
        loaded = self._get_file(file_id)
        if not loaded:
            return {"error": f"File not found: {file_id}"}

        df = loaded.data

        # Limit count
        count = min(count, 20)

        # Slice rows
        end_row = min(start_row + count, len(df))
        sample_df = df.iloc[start_row:end_row]

        # Select columns if specified
        if columns:
            valid_cols = [c for c in columns if c in df.columns]
            if valid_cols:
                sample_df = sample_df[valid_cols]

        # Convert to records
        records = sample_df.to_dict(orient="records")

        return {
            "rows": records,
            "start_row": start_row,
            "end_row": end_row,
            "total_rows": len(df),
        }
