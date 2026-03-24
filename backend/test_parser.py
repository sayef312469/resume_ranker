# test_parser.py — run this to test CV parsing
import asyncio
from cv_parser import (
    extract_from_pdf,
    extract_from_docx,
    extract_from_txt,
    validate_cv_text,
    preview_text
)

# Test with a sample PDF
async def test():
    # Test TXT
    sample_text = b"John Doe\nExperience: 5 years Python\nSkills: ML, NLP"
    text = extract_from_txt(sample_text)
    print("TXT extraction:")
    print(text)

    # Validate
    validation = validate_cv_text(text)
    print(f"\nValidation: {validation}")

    # Preview
    print(f"\nPreview: {preview_text(text, 10)}")

asyncio.run(test())