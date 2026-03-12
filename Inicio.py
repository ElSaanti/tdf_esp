import streamlit as st 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.set_page_config(page_title="Demo TF-IDF en Español", layout="wide")

st.title("Demo TF-IDF en Español")
st.caption("Modo Detective Semántico: descubre qué documento se parece más a tu pregunta.")

# Documentos de ejemplo
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    # Minúsculas
    text = text.lower()
    # Solo letras españolas y espacios
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    # Tokenizar
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def detective_message(score):
    if score > 0.6:
        return "Caso resuelto con alta precisión."
    elif score > 0.25:
        return "Hay una pista bastante clara."
    elif score > 0.01:
        return "Coincidencia encontrada, pero con dudas."
    else:
        return "El caso está borroso, la evidencia es débil."

# Layout en dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("Documentos (uno por línea):", default_docs, height=150)
    question = st.text_input("Escribe tu pregunta:", "¿Dónde juegan el perro y el gato?")

with col2:
    st.markdown("### Preguntas sugeridas")
    
    if st.button("¿Dónde juegan el perro y el gato?", use_container_width=True):
        st.session_state.question = "¿Dónde juegan el perro y el gato?"
        st.rerun()
    
    if st.button("¿Qué hacen los niños en el parque?", use_container_width=True):
        st.session_state.question = "¿Qué hacen los niños en el parque?"
        st.rerun()
        
    if st.button("¿Cuándo cantan los pájaros?", use_container_width=True):
        st.session_state.question = "¿Cuándo cantan los pájaros?"
        st.rerun()
        
    if st.button("¿Dónde suena la música alta?", use_container_width=True):
        st.session_state.question = "¿Dónde suena la música alta?"
        st.rerun()
        
    if st.button("¿Qué animal maúlla durante la noche?", use_container_width=True):
        st.session_state.question = "¿Qué animal maúlla durante la noche?"
        st.rerun()

    st.markdown("---")
    st.markdown("### Detalle especial")
    st.info("Este analizador compara tu pregunta con cada documento usando similitud semántica basada en TF-IDF.")

# Actualizar pregunta si se seleccionó una sugerida
if 'question' in st.session_state:
    question = st.session_state.question

if st.button("Analizar", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("Ingresa al menos un documento.")
    elif not question.strip():
        st.error("Escribe una pregunta.")
    else:
        # Crear vectorizador TF-IDF
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1
        )
        
        # Ajustar con documentos
        X = vectorizer.fit_transform(documents)
        
        # Mostrar matriz TF-IDF
        st.markdown("### Matriz TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
        
        # Calcular similitud con la pregunta
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        
        # Encontrar mejor respuesta
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        # Ranking completo
        ranking_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        }).sort_values(by="Similitud", ascending=False).reset_index(drop=True)
        
        # Mostrar respuesta
        st.markdown("### Respuesta")
        st.markdown(f"**Tu pregunta:** {question}")
        
        if best_score > 0.01:
            st.success(f"**Respuesta:** {best_doc}")
            st.info(f"Similitud: {best_score:.3f}")
        else:
            st.warning(f"**Respuesta (baja confianza):** {best_doc}")
            st.info(f"Similitud: {best_score:.3f}")

        # Toque único y divertido
        st.markdown("### Modo Detective Semántico")
        st.progress(min(float(best_score), 1.0))
        st.write(detective_message(best_score))

        # Ranking visual
        st.markdown("### Ranking de coincidencias")
        st.dataframe(ranking_df.round(3), use_container_width=True)

        # Mostrar palabras más importantes del mejor documento
        st.markdown("### Pistas clave del mejor documento")
        feature_names = vectorizer.get_feature_names_out()
        best_doc_vector = X[best_idx].toarray().flatten()
        top_indices = best_doc_vector.argsort()[::-1][:5]
        top_words = [(feature_names[i], best_doc_vector[i]) for i in top_indices if best_doc_vector[i] > 0]

        if top_words:
            pistas_df = pd.DataFrame(top_words, columns=["Término", "Peso TF-IDF"])
            st.table(pistas_df.round(3))
        else:
            st.write("No se encontraron términos destacados.")
