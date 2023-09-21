package com.tjjhtjh.memorise.domain.memo.exception;

public class BookMarkException extends NullPointerException{

    private String message;

    public BookMarkException(String message) {
        this.message = message;
    }
}
