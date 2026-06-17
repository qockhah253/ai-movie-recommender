"""CustomTkinter desktop interface for the movie recommender."""

from __future__ import annotations

import customtkinter as ctk

from .model import RecommenderModel

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class MovieApp(ctk.CTk):
    """Main application window."""

    def __init__(self, recommender: RecommenderModel):
        super().__init__()
        self.recommender = recommender
        self.data = recommender.data

        self.title("AI Movie Recommender System (Linear Regression)")
        self.geometry("950x650")
        self.minsize(820, 560)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()

    # ------------------------------------------------------------------ UI --
    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self, width=260)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        sidebar.grid_propagate(False)

        ctk.CTkLabel(
            sidebar,
            text="BỘ LỌC PHIM",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(pady=20)

        self._label(sidebar, "Thời lượng (phút)")
        self.runtime_entry = self._entry(sidebar, "120")

        self._label(sidebar, "Năm phát hành")
        self.year_entry = self._entry(sidebar, "2024")

        self._label(sidebar, "Thể loại")
        self.genre_box = ctk.CTkComboBox(
            sidebar, values=self.data.genres, state="readonly"
        )
        self.genre_box.pack(fill="x", padx=20, pady=10)
        self.genre_box.set(self.data.genres[0])

        self._label(sidebar, "Quốc gia")
        self.country_box = ctk.CTkComboBox(
            sidebar, values=self.data.countries, state="readonly"
        )
        self.country_box.pack(fill="x", padx=20, pady=10)
        self.country_box.set(self.data.countries[0])

        ctk.CTkButton(
            sidebar,
            text="GỢI Ý PHIM",
            height=45,
            fg_color="#1f6aa5",
            hover_color="#144870",
            command=self.process,
        ).pack(fill="x", padx=20, pady=30)

        m = self.recommender.best_metrics
        n_movies = len(self.recommender.data.df)
        ctk.CTkLabel(
            sidebar,
            text=(
                f"Mô hình tốt nhất: {self.recommender.best_name}\n"
                f"R\u00b2 = {m['r2']:.3f}  |  MAE = {m['mae']:.3f}\n"
                f"Dữ liệu: {n_movies:,} phim"
            ),
            font=ctk.CTkFont(size=11),
            text_color="gray70",
            justify="left",
        ).pack(side="bottom", pady=15)

    def _build_main(self) -> None:
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        ctk.CTkLabel(
            main,
            text="Danh sách phim gợi ý",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(pady=10)

        self.score_label = ctk.CTkLabel(
            main,
            text="Điểm dự báo: --",
            text_color="#3bb2fe",
            font=ctk.CTkFont(size=18),
        )
        self.score_label.pack()

        self.result_box = ctk.CTkTextbox(
            main, font=("Segoe UI", 15), border_width=2
        )
        self.result_box.pack(fill="both", expand=True, pady=20)

    def _label(self, parent, text: str) -> None:
        ctk.CTkLabel(
            parent, text=text, font=ctk.CTkFont(size=13)
        ).pack(anchor="w", padx=25)

    def _entry(self, parent, placeholder: str):
        entry = ctk.CTkEntry(parent, placeholder_text=placeholder)
        entry.pack(fill="x", padx=20, pady=10)
        return entry

    # ------------------------------------------------------------- actions --
    def process(self) -> None:
        try:
            runtime = float(self._value(self.runtime_entry))
            year = int(self._value(self.year_entry))
        except ValueError:
            self._show_error("Vui lòng nhập thời lượng và năm hợp lệ (dạng số).")
            return

        genre = self.genre_box.get()
        country = self.country_box.get()

        rating = self.recommender.predict_rating(runtime, year, genre, country)
        self.score_label.configure(
            text=f"Dự báo điểm đánh giá (Linear Regression): {rating:.2f} / 10"
        )

        recommendations = self.recommender.recommend(runtime, year, genre, country)
        self._show_recommendations(recommendations)

    @staticmethod
    def _value(entry) -> str:
        """Return the entry's text, falling back to its placeholder."""
        return entry.get().strip() or entry.cget("placeholder_text")

    def _show_recommendations(self, recommendations) -> None:
        lines = [
            " CÓ THỂ BẠN SẼ THÍCH CÁC PHIM SAU",
            " " + "\u2501" * 36,
            "",
        ]
        for _, movie in recommendations.iterrows():
            lines.append(f" ● {str(movie['title']).upper()}")
            lines.append(
                f"     Năm: {int(movie['release_year'])}  |  "
                f"Quốc gia: {movie['production_countries']}"
            )
            lines.append(
                f"     Thể loại: {movie['genres']}  |  "
                f"Điểm thực tế: {movie['vote_average']}/10"
            )
            lines.append("")

        self.result_box.delete("1.0", ctk.END)
        self.result_box.insert("1.0", "\n".join(lines))

    def _show_error(self, message: str) -> None:
        self.result_box.delete("1.0", ctk.END)
        self.result_box.insert("1.0", f"⚠  {message}")
