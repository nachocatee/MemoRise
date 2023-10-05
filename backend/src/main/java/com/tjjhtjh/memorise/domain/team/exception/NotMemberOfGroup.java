package com.tjjhtjh.memorise.domain.team.exception;

public class NotMemberOfGroup extends RuntimeException {
    public NotMemberOfGroup(String message) {
        super(message);
    }

    public NotMemberOfGroup(String message, Throwable cause) {
        super(message, cause);
    }
}
