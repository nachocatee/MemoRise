package com.tjjhtjh.memorise.domain.memo.exception;

import org.springframework.data.crossstore.ChangeSetPersister;

public class MemoException extends ChangeSetPersister.NotFoundException {
    private String message;

    public MemoException(String message) {
        this.message = message;
    }


}
