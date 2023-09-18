package com.tjjhtjh.memorise.domain.user.exception;

public class NoUserException extends RuntimeException {
    public NoUserException(String message) {
        super(message);
    }

    public NoUserException(String message, Throwable cause) {
        super(message, cause);
    }
}
