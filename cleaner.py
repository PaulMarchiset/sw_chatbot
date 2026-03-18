import os
import re
from pathlib import Path

# Un seul dossier : on va travailler directement dedans !
DOSSIER_CORPUS = "corpus_starwars"

def nettoyer_texte(texte):
    """Nettoie le texte brut de Wookieepedia pour l'IA."""
    
    # --- 1. COUPE AU SCALPEL (La fin du fichier) ---
    mots_de_coupure = [
        "Appearances[]", 
        "Sources[]", 
        "Notes and references[]",
        "External links[]", 
        "Behind the scenes[]", 
        "In other languages"
    ]
    for mot in mots_de_coupure:
        if mot in texte:
            texte = texte.split(mot)[0] 

    # --- 2. FILTRAGE DES DÉCHETS (Le début du fichier) ---
    lignes = texte.split('\n')
    lignes_propres = []
    
    for ligne in lignes:
        ligne_strip = ligne.strip()
        
        if not ligne_strip or ligne_strip == "[]" or ligne_strip == "[Source]":
            continue
            
        if ligne_strip.startswith("This article"): continue
        if ligne_strip.startswith("It is requested that"): continue
        if ligne_strip.startswith("Please help Wookieepedia"): continue
        if ligne_strip.startswith("Parts of this article"): continue
        if ligne_strip.startswith("For other uses, see"): continue
        if ligne_strip.startswith("There are two conflicting sources"): continue
        if ligne_strip.startswith("Lucasfilm has not established"): continue
        if ligne_strip.startswith("Contents"): continue 

        lignes_propres.append(ligne)

    texte_propre = '\n'.join(lignes_propres)

    # --- 3. POLISSAGE FINAL ---
    texte_propre = re.sub(r'\n{3,}', '\n\n', texte_propre)

    return texte_propre.strip()

def main():
    print(f"⚠️ DANGER : Mode 'Écrasement' activé. Nettoyage de '{DOSSIER_CORPUS}'...")
    
    chemin_source = Path(DOSSIER_CORPUS)
    if not chemin_source.exists():
        print(f"❌ Erreur : Le dossier {DOSSIER_CORPUS} n'existe pas.")
        return

    fichiers_traites = 0
    
    # On fouille dans tous les sous-dossiers
    for chemin_fichier in chemin_source.rglob("*.txt"):
        try:
            # 1. On lit le texte sale
            texte_sale = chemin_fichier.read_text(encoding="utf-8", errors="ignore")
            
            # 2. On le nettoie en mémoire
            texte_propre = nettoyer_texte(texte_sale)
            
            # 3. ON ÉCRASE le fichier original avec le texte propre
            chemin_fichier.write_text(texte_propre, encoding="utf-8")
            fichiers_traites += 1
            
        except Exception as e:
            print(f"⚠️ Erreur lors du traitement de {chemin_fichier.name} : {e}")

    print("-" * 40)
    print(f"✅ Purge terminée ! {fichiers_traites} fichiers ont été nettoyés et écrasés.")

if __name__ == "__main__":
    main()