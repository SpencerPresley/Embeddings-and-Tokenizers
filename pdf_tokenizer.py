from __future__ import annotations

from langchain_community.document_loaders import PyMuPDFLoader
import numpy as np
from typing import List, Dict, Union, Optional
import os
from tokenizer import TiktokenWrapper
import argparse


class PDFTokenizer:
    """
    A class for extracting text from PDF files and tokenizing it using tiktoken.
    """

    def __init__(self, encoding_name: str | None = "cl100k_base"):
        """
        Initialize the PDF tokenizer.

        Args:
            encoding_name (str): The encoding to use for tokenization (default: cl100k_base)
        """
        self.tokenizer = TiktokenWrapper(encoding_name)
        self.pdf_path = None
        self.pages = []
        self.text = ""

    def load_pdf(self, pdf_path: str) -> bool:
        """
        Load a PDF file and extract its text content.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(pdf_path):
            print(f"Error: File not found - {pdf_path}")
            return False

        if not pdf_path.lower().endswith(".pdf"):
            print(f"Error: File is not a PDF - {pdf_path}")
            return False

        try:
            loader = PyMuPDFLoader(pdf_path)
            self.pages = loader.load()
            self.pdf_path = pdf_path
            self.text = self._extract_full_text()
            return True
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return False

    def _extract_full_text(self) -> str:
        """
        Extract and concatenate all text from the loaded PDF pages.

        Returns:
            Combined text from all pages
        """
        return "\n".join([page.page_content for page in self.pages])

    def get_page_count(self) -> int:
        """
        Get the number of pages in the loaded PDF.

        Returns:
            Number of pages
        """
        return len(self.pages)

    def get_page_text(self, page_num: int) -> str:
        """
        Get text from a specific page.

        Args:
            page_num: The 0-indexed page number

        Returns:
            Text content of the page
        """
        if not self.pages or page_num < 0 or page_num >= len(self.pages):
            return ""
        return self.pages[page_num].page_content

    def tokenize_full_document(
        self, as_numpy: bool | None = False
    ) -> Union[List[int], np.ndarray]:
        """
        Tokenize the entire document.

        Args:
            as_numpy (bool): Whether to return tokens as a numpy array

        Returns:
            List or numpy array of token IDs for the entire document
        """
        return self.tokenizer.encode(self.text, as_numpy=as_numpy)

    def tokenize_page(
        self, page_num: int, as_numpy: bool | None = False
    ) -> Union[List[int], np.ndarray]:
        """
        Tokenize a specific page.

        Args:
            page_num (int): The 0-indexed page number
            as_numpy (bool): Whether to return tokens as a numpy array

        Returns:
            List or numpy array of token IDs for the specified page
        """
        page_text = self.get_page_text(page_num)
        return self.tokenizer.encode(page_text, as_numpy=as_numpy)

    def get_token_count_by_page(self) -> Dict[int, int]:
        """
        Count tokens in each page of the document.

        Returns:
            Dictionary mapping page numbers to token counts
        """
        token_counts = {}
        for i in range(len(self.pages)):
            page_text = self.get_page_text(i)
            token_counts[i] = self.tokenizer.count_tokens(page_text)
        return token_counts

    def get_total_token_count(self) -> int:
        """
        Get the total token count for the entire document.

        Returns:
            Total number of tokens
        """
        return self.tokenizer.count_tokens(self.text)

    def save_token_report(
        self,
        output_file: str | None = "token_report.txt",
        include_decoded: bool | None = True,
        tokens_per_line: int | None = 10,
        max_decoded_length: int | None = 20,
    ) -> None:
        """
        Generate and save a detailed token report to a file.

        Args:
            output_file (str): Path to save the report
            include_decoded (bool): Whether to include decoded token text
            tokens_per_line (int): Number of tokens to display per line
            max_decoded_length (int): Maximum length for decoded token text
        """
        page_count = self.get_page_count()
        total_tokens = self.get_total_token_count()

        with open(output_file, "w", encoding="utf-8") as f:
            # Write header with summary information
            f.write(f"{'=' * 80}\n")
            f.write(f"TOKEN ANALYSIS REPORT\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"PDF: {self.pdf_path}\n")
            f.write(f"Encoding: {self.tokenizer.encoding_name}\n")
            f.write(f"Total Pages: {page_count}\n")
            f.write(f"Total Tokens: {total_tokens}\n\n")

            # Token distribution table
            f.write(f"{'=' * 80}\n")
            f.write(f"TOKEN DISTRIBUTION BY PAGE\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"{'Page #':<10}{'Token Count':<15}{'% of Total':<15}\n")
            f.write(f"{'-' * 40}\n")

            token_counts = self.get_token_count_by_page()
            for page_num, count in token_counts.items():
                percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
                f.write(f"{page_num + 1:<10}{count:<15}{percentage:.2f}%\n")

            # Detailed token analysis by page
            f.write(f"\n{'=' * 80}\n")
            f.write(f"DETAILED TOKEN ANALYSIS BY PAGE\n")
            f.write(f"{'=' * 80}\n\n")

            for page_num in range(page_count):
                page_tokens = self.tokenize_page(page_num)
                f.write(f"\n{'*' * 80}\n")
                f.write(f"PAGE {page_num + 1}: {len(page_tokens)} TOKENS\n")
                f.write(f"{'*' * 80}\n\n")

                # Format tokens in groups for readability
                for i in range(0, len(page_tokens), tokens_per_line):
                    chunk = page_tokens[i : i + tokens_per_line]

                    # Token IDs
                    token_str = ", ".join(f"{token:5d}" for token in chunk)
                    f.write(
                        f"Tokens [{i:4d}-{min(i + tokens_per_line, len(page_tokens)) - 1:4d}]: {token_str}\n"
                    )

                    # Decoded tokens if requested
                    if include_decoded:
                        decoded_tokens = []
                        for token in chunk:
                            # Decode individual token and sanitize for display
                            decoded = self.tokenizer.decode([token])
                            # Replace newlines and limit length
                            decoded = decoded.replace("\n", "\\n").replace("\r", "\\r")
                            if len(decoded) > max_decoded_length:
                                decoded = decoded[: max_decoded_length - 3] + "..."
                            # Handle empty strings (show as special placeholder)
                            if decoded == "":
                                decoded = (
                                    "␣"  # Use special character for whitespace/empty
                                )
                            decoded_tokens.append(f"'{decoded}'")

                        decoded_str = ", ".join(decoded_tokens)
                        f.write(
                            f"Decoded [{i:4d}-{min(i + tokens_per_line, len(page_tokens)) - 1:4d}]: {decoded_str}\n\n"
                        )

            f.write(f"\n{'=' * 80}\n")
            f.write(f"END OF REPORT\n")
            f.write(f"{'=' * 80}\n")


def add_parser_args(parser: argparse.ArgumentParser) -> None:
    """Add command line arguments to the parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add arguments to
    """
    parser.add_argument("pdf_path", help="Path to the PDF file to analyze")
    parser.add_argument(
        "--encoding", default="cl100k_base", help="Tiktoken encoding to use"
    )
    parser.add_argument(
        "--output", default="token_report.txt", help="Output file for the token report"
    )
    parser.add_argument(
        "--no-decode",
        action="store_false",
        dest="include_decoded",
        help="Disable token decoding in the report",
    )
    parser.add_argument(
        "--tokens-per-line",
        type=int,
        default=10,
        help="Number of tokens to display per line",
    )


# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze token usage in PDF documents")
    add_parser_args(parser)

    args = parser.parse_args()

    # Initialize the PDF tokenizer
    pdf_tokenizer = PDFTokenizer(encoding_name=args.encoding)

    # Load the PDF
    if pdf_tokenizer.load_pdf(args.pdf_path):
        # Get basic info
        page_count = pdf_tokenizer.get_page_count()
        total_tokens = pdf_tokenizer.get_total_token_count()

        print(f"PDF: {args.pdf_path}")
        print(f"Pages: {page_count}")
        print(f"Total tokens: {total_tokens}")

        # Generate and save the token report
        pdf_tokenizer.save_token_report(
            output_file=args.output,
            include_decoded=args.include_decoded,
            tokens_per_line=args.tokens_per_line,
        )

        print(f"\nToken report saved to: {args.output}")
    else:
        print("Failed to load PDF file.")
