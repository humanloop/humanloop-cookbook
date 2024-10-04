def count_trace_children(log):
    """Assert all logs in Trace have the trace_children key and count them"""

    def sub_fn(log):
        logs = 0
        assert "trace_children" in log, f"Error: {log}"
        for child in log["trace_children"]:
            logs += count_trace_children(child)
        return logs + len(log["trace_children"])

    return sub_fn(log)
