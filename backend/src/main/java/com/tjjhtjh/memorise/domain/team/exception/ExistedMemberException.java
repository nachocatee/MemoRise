package com.tjjhtjh.memorise.domain.team.exception;

public class ExistedMemberException extends RuntimeException {
    public ExistedMemberException(String message) {
        super(message);
    }

    public ExistedMemberException(String message, Throwable cause) {
        super(message, cause);
    }
}
