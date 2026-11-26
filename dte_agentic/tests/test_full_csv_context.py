"""Offline regression checks: python -m unittest discover -s tests -v.

Extract pure app functions to avoid starting Flask, Chroma or remote models.
"""
import ast
import logging
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

APP = Path(__file__).resolve().parents[1] / 'app.py'


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class FullCSVContextTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(APP.read_text())
        functions = [node for node in self.tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {'load_csv_documents', 'load_chunks',
                                       'normalize_text', 'retrieve_documents'}]
        self.store = Mock()
        self.ns = dict(Document=Document, app=SimpleNamespace(logger=logging.getLogger(__name__)),
                       vectorstore=self.store, TOP_K=8, RAG_MAX_DISTANCE=0.6,
                       CHUNK_SIZE=1200, re=re)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(APP), 'exec'), self.ns)
        self.csv_path = APP.parent / 'data' / 'data.csv'
        self.csv_docs = self.ns['load_csv_documents']([self.csv_path])
        self.ns['csv_documents'] = self.csv_docs

    def test_supplied_csv_is_complete_and_unchanged(self):
        expected = self.csv_path.read_bytes().decode('utf-8-sig')
        self.assertGreater(len(expected), 4_000_000)
        self.assertEqual(len(self.csv_docs), 1)
        self.assertEqual(self.csv_docs[0].page_content, expected)
        self.assertEqual(self.csv_docs[0].metadata['source'], str(self.csv_path))

    def test_full_csv_survives_empty_or_rejected_pdf_results(self):
        for candidates in ([], [(Document('unrelated', {'source': 'paper.pdf'}), 0.9)]):
            self.store.similarity_search_with_score.return_value = candidates
            result = self.ns['retrieve_documents']({'input': 'question'})
            self.assertEqual(result, self.csv_docs)

    def test_pdf_selection_and_complete_csv_coexist(self):
        self.store.similarity_search_with_score.return_value = [
            (Document('passage B', {'source': 'b.pdf', 'page': 0}), 0.5),
            (Document('passage A', {'source': 'a.pdf', 'page': 0}), 0.2),
            (Document('passage A', {'source': 'a.pdf', 'page': 0}), 0.3),
            (Document('excluded', {'source': 'c.pdf'}), 0.8),
        ]
        result = self.ns['retrieve_documents']({'input': 'question plus live data',
                                                'retrieval_query': 'question'})
        self.assertEqual([d.page_content for d in result[:2]], ['passage A', 'passage B'])
        self.assertIs(result[-1], self.csv_docs[0])
        self.store.similarity_search_with_score.assert_called_once_with('question', k=8)

    def test_multiple_csvs_preserve_quoted_fields_and_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('a.csv', 'b.CSV')]
            content = 'name,value\r\n"a, b","line one\r\nline two"\r\nlast,42\r\n'
            for path in paths:
                path.write_bytes(('\ufeff' + content).encode('utf-8'))
            docs = self.ns['load_csv_documents'](paths)
            self.assertEqual([d.page_content for d in docs], [content, content])
            self.ns['csv_documents'] = docs
            self.store.similarity_search_with_score.return_value = []
            self.assertEqual(self.ns['retrieve_documents']({'input': 'question'}), docs)

    def test_csv_cannot_be_chunked(self):
        with self.assertRaises(ValueError):
            self.ns['load_chunks'](self.csv_path)

    def test_pdf_chunking_preserves_page_metadata(self):
        self.ns['PdfReader'] = lambda path: SimpleNamespace(pages=[
            SimpleNamespace(extract_text=lambda: 'A' * 1300),
            SimpleNamespace(extract_text=lambda: 'B' * 100),
        ])
        docs = self.ns['load_chunks'](Path('paper.pdf'))
        self.assertEqual([len(d.page_content) for d in docs], [1200, 201])
        self.assertEqual(docs[1].metadata['page'], 0)
        self.assertEqual(docs[1].metadata['page_end'], 1)


if __name__ == '__main__':
    unittest.main()
