import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.set_page_config(page_title="Demo TF-IDF", page_icon="📚", layout="wide")

st.title("📚 Buscador TF-IDF en Español")
st.caption("Consulta documentos y encuentra el texto más similar a tu pregunta.")


default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""


stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems


if "question" not in st.session_state:
    st.session_state.question = "¿Dónde juegan el perro y el gato?"


col1, col2 = st.columns([2.3, 1])

with col1:
    text_input = st.text_area(
        "📝 Documentos (uno por línea):",
        value=default_docs,
        height=180,
        placeholder="Escribe aquí varios documentos, uno por cada línea"
    )

    question = st.text_input(
        "❓ Escribe tu pregunta:",
        value=st.session_state.question
    )

with col2:
    st.markdown("### Preguntas sugeridas")

    suggested_questions = [
        "¿Dónde juegan el perro y el gato?",
        "¿Qué hacen los niños en el parque?",
        "¿Cuándo cantan los pájaros?",
        "¿Dónde suena la música alta?",
        "¿Qué animal maúlla durante la noche?"
    ]

    for q in suggested_questions:
        if st.button(q, use_container_width=True):
            st.session_state.question = q
            st.rerun()

if st.button("🔍 Analizar documentos", type="primary", use_container_width=True):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) == 0:
        st.error("Ingresa al menos un documento.")
    elif not question.strip():
        st.error("Escribe una pregunta antes de analizar.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1
        )

        X = vectorizer.fit_transform(documents)

        st.markdown("### 📊 Matriz TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)

        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()

        results_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        }).sort_values(by="Similitud", ascending=False)

        best_row = results_df.iloc[0]

        st.markdown("### 🎯 Mejor coincidencia")
        st.markdown(f"**Pregunta:** {question}")

        if best_row["Similitud"] > 0.01:
            st.success(f"**Respuesta encontrada:** {best_row['Texto']}")
            st.info(f"Similitud: {best_row['Similitud']:.3f}")
        else:
            st.warning(f"**Coincidencia de baja confianza:** {best_row['Texto']}")
            st.info(f"Similitud: {best_row['Similitud']:.3f}")

        st.markdown("### 📌 Ranking de documentos")
        st.dataframe(results_df.round(3), use_container_width=True)
