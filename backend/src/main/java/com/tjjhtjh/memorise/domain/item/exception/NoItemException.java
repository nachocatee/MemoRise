package com.tjjhtjh.memorise.domain.item.exception;

public class NoItemException extends NullPointerException{

    private String message;

    public NoItemException(String message) {
        this.message = message;
    }
}
