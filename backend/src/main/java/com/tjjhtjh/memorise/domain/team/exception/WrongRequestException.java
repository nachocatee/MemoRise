package com.tjjhtjh.memorise.domain.team.exception;

public class WrongRequestException extends RuntimeException {
    public WrongRequestException(String message) {
        super(message);
    }

    public WrongRequestException(String message, Throwable cause) {
        super(message, cause);
    }
}
