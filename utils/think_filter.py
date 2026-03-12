"""Streaming filter for <think>...</think> tags across SSE chunk boundaries."""


class ThinkTagFilter:
    """
    Streaming filter that safely removes <think>...</think> blocks even when
    tags are split across chunk boundaries.

    Usage:
        filter = ThinkTagFilter()
        for chunk in incoming_chunks:
            display_text = filter.feed(chunk)
            # display_text has <think> blocks removed
        final_text = filter.flush()  # flush any remaining buffer
    """

    def __init__(self):
        self._inside_think = False
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        """
        Feed a chunk, return filtered text ready for display.

        Args:
            chunk: New text chunk from the stream

        Returns:
            Filtered text with complete <think>...</think> blocks removed.
            Partial tags at chunk boundaries are kept in buffer.
        """
        self._buffer += chunk
        output = []

        while self._buffer:
            if self._inside_think:
                # Looking for closing tag
                end = self._buffer.find("</think>")
                if end == -1:
                    # Closing tag not found, check if we have a partial tag at the end
                    if self._is_partial_closing_tag(self._buffer):
                        # Keep the partial tag in buffer, wait for more
                        break
                    # No partial tag, discard everything until next attempt
                    self._buffer = ""
                    break
                # Found closing tag, skip it and everything before
                self._inside_think = False
                self._buffer = self._buffer[end + 8:]
            else:
                # Looking for opening tag
                start = self._buffer.find("<think>")
                if start == -1:
                    # Opening tag not found, check for partial tag at end
                    partial_len = self._get_partial_opening_tag_len(self._buffer)
                    if partial_len > 0:
                        # Output everything except the partial tag
                        output.append(self._buffer[:-partial_len])
                        self._buffer = self._buffer[-partial_len:]
                    else:
                        # No partial tag, output everything and clear buffer
                        output.append(self._buffer)
                        self._buffer = ""
                    break
                # Found opening tag, output text before it
                output.append(self._buffer[:start])
                self._inside_think = True
                self._buffer = self._buffer[start + 7:]

        return "".join(output)

    def flush(self) -> str:
        """
        Flush remaining buffer at end of stream.

        Returns:
            Any remaining text in buffer. If we're still inside a <think> block,
            returns empty string (discarding the incomplete block).
        """
        result = self._buffer if not self._inside_think else ""
        self._buffer = ""
        self._inside_think = False
        return result

    def _is_partial_closing_tag(self, text: str) -> bool:
        """Check if text ends with a partial </think> tag."""
        closing_tag = "</think>"
        # Check all possible partial endings from 1 to 7 chars
        for i in range(1, len(closing_tag)):
            if text.endswith(closing_tag[:i]):
                return True
        return False

    def _get_partial_opening_tag_len(self, text: str) -> int:
        """
        Get length of partial <think> tag at end of text.
        Returns 0 if no partial tag found.
        """
        opening_tag = "<think>"
        # Check all possible partial endings from 1 to 6 chars
        for i in range(1, len(opening_tag)):
            if text.endswith(opening_tag[:i]):
                return i
        return 0
