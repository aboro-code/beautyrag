from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Numeric, SmallInteger, Text, TIMESTAMP, func
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings


class Base(DeclarativeBase):
    pass


class Product(Base):
    """A skincare/beauty product: catalog fields + vector embedding + full-text index."""

    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(Text, primary_key=True)
    product_name: Mapped[str] = mapped_column(Text, nullable=False)
    brand_name: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(Text)
    price_usd: Mapped[float | None] = mapped_column(Numeric)
    rating: Mapped[float | None] = mapped_column(Numeric)
    reviews_count: Mapped[int | None] = mapped_column(SmallInteger)
    ingredients: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)

    embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.embedding_dim))
    # Generated column; created via raw DDL in data/schema.sql, mapped read-only here.
    search_vector: Mapped[str | None] = mapped_column(TSVECTOR)

    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    reviews: Mapped[list["Review"]] = relationship(back_populates="product")


class Review(Base):
    """A single customer review tied to a product, used as grounding context for /ask."""

    __tablename__ = "reviews"

    review_id: Mapped[str] = mapped_column(Text, primary_key=True)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.product_id", ondelete="CASCADE"))
    author_id: Mapped[str | None] = mapped_column(Text)
    rating: Mapped[int | None] = mapped_column(SmallInteger)
    review_title: Mapped[str | None] = mapped_column(Text)
    review_text: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    product: Mapped["Product"] = relationship(back_populates="reviews")
