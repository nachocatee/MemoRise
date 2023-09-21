package com.tjjhtjh.memorise.domain.memo.exception;

public class BookmarkException extends NullPointerException{

    private String message;

    public BookmarkException(String message) {
        this.message = message;
    }
}
