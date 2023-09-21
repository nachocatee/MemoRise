package com.tjjhtjh.memorise.domain.memo.service.dto.request;

import com.tjjhtjh.memorise.domain.memo.repository.entity.AccessType;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
public class MemoRequest {

    private String content;
    private Long itemSeq;
    private AccessType accessType;
    private User user;
    private Long userId;

    public Memo registToEntity(User addUser) {
        return Memo.builder()
                .content(content)
                .itemSeq(itemSeq)
                .accessType(accessType)
                .user(addUser)
                .build();
    }

    public Memo updateToEntity(Long memoId,MemoRequest memoRequest,User user){
        return Memo.builder()
                .memoSeq(memoId)
                .content(memoRequest.getContent())
                .itemSeq(memoRequest.getItemSeq())
                .accessType(memoRequest.getAccessType())
                .user(user)
                .isDeleted(0)
                .build();
    }

    public Memo deleteToEntity(Memo memo,User writeUser){
        return Memo.builder()
                .memoSeq(memo.getMemoSeq())
                .isDeleted(1)
                .itemSeq(memo.getItemSeq())
                .accessType(memo.getAccessType())
                .content(memo.getContent())
                .deletedAt(LocalDateTime.now())
                .user(writeUser)
                .build();
    }

}
