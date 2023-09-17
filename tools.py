import os
from io import BytesIO
import re
from typing import Any, Dict, List


import streamlit as st
import pandas as pd

from pypdf import PdfReader
import docx2txt

def pretty_print_docs(docs):
    st.markdown(f"\n\n".join([f"## Document {i+1}: {d.metadata['source'].split('/')[-1]} \n\n" + "Page " + str(d.metadata['page']) + "\n\n" + d.page_content for i, d in enumerate(docs)]))
