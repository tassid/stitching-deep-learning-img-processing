#!/usr/bin/env python3
"""
Verifica se todas as citações nos arquivos LaTeX existem no BibTeX
"""

import re
import os
from pathlib import Path

def extract_citations_from_tex(tex_file):
    """Extrai todas as citacoes de um arquivo .tex"""
    citations = set()
    try:
        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Encontra todos os \cite{...} e \citep{...} e \citet{...}
            patterns = [
                r'\\cite\{([^}]+)\}',
                r'\\citep\{([^}]+)\}',
                r'\\citet\{([^}]+)\}',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    # Pode ter múltiplas referências separadas por vírgula
                    refs = [r.strip() for r in match.split(',')]
                    citations.update(refs)
    except Exception as e:
        print(f"Erro ao ler {tex_file}: {e}")
    return citations

def extract_bibtex_keys(bib_file):
    """Extrai todas as chaves de entrada do BibTeX"""
    keys = set()
    try:
        with open(bib_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Encontra todas as entradas @tipo{chave,
            matches = re.findall(r'@\w+\{([^,]+),', content)
            keys.update(matches)
    except Exception as e:
        print(f"Erro ao ler {bib_file}: {e}")
    return keys

def main():
    base_dir = Path(r"C:\Users\TASSI\Documents\GitHub\stitching-deep-learning-img-processing\TCC_TASSIANE_2025_NOVOTEMPLATE")

    # Encontra todos os arquivos .tex
    tex_files = []
    tex_files.append(base_dir / "main.tex")
    capitulos_dir = base_dir / "capitulos"
    if capitulos_dir.exists():
        tex_files.extend(capitulos_dir.glob("*.tex"))

    # Extrai todas as citações
    all_citations = set()
    for tex_file in tex_files:
        if tex_file.exists():
            citations = extract_citations_from_tex(tex_file)
            if citations:
                print(f"\n{tex_file.name}: {len(citations)} citações únicas")
            all_citations.update(citations)

    print(f"\n{'='*60}")
    print(f"Total de citações únicas encontradas: {len(all_citations)}")

    # Extrai chaves do BibTeX
    bib_file = base_dir / "main.bib"
    bibtex_keys = extract_bibtex_keys(bib_file)
    print(f"Total de entradas no BibTeX: {len(bibtex_keys)}")

    # Verifica citações faltantes
    missing = all_citations - bibtex_keys

    print(f"\n{'='*60}")
    if missing:
        print(f"ATENCAO: {len(missing)} referencias FALTANDO no BibTeX:")
        for ref in sorted(missing):
            print(f"  - {ref}")
    else:
        print("OK - Todas as citacoes estao presentes no BibTeX!")

    # Verifica entradas não utilizadas
    unused = bibtex_keys - all_citations
    if unused:
        print(f"\n{len(unused)} entradas no BibTeX NAO citadas no texto:")
        for ref in sorted(list(unused)[:10]):  # Mostra so as primeiras 10
            print(f"  - {ref}")
        if len(unused) > 10:
            print(f"  ... e mais {len(unused) - 10} entradas nao citadas")

    print(f"\n{'='*60}")
    print("\nCitacoes encontradas por arquivo:")
    for tex_file in sorted(tex_files):
        if tex_file.exists():
            citations = extract_citations_from_tex(tex_file)
            if citations:
                print(f"\n{tex_file.name}:")
                for cite in sorted(citations):
                    status = "OK" if cite in bibtex_keys else "X FALTANDO"
                    print(f"  {status} {cite}")

if __name__ == "__main__":
    main()
