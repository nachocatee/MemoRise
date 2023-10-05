package com.tjjhtjh.memorise.domain.team.exception;

public class NoTeamException extends RuntimeException {
    public NoTeamException(String message) {
        super(message);
    }

    public NoTeamException(String message, Throwable cause) {
        super(message, cause);
    }
}
