package com.tjjhtjh.memorise.domain.team.exception;

public class WrongCodeException extends RuntimeException {
    public WrongCodeException(String message) {
        super(message);
    }

    public WrongCodeException(String message, Throwable cause) {
        super(message, cause);
    }
}
