package com.tjjhtjh.memorise.domain.memo.service.dto.request;

import com.tjjhtjh.memorise.domain.item.repository.entity.Item;
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
    private Item item;
    private AccessType accessType;
    private User user;
    private Long userId;

    public Memo registToEntity(User addUser, Item item1) {
        return Memo.builder()
                .content(content)
                .item(item1)
                .accessType(accessType)
                .user(addUser)
                .build();
    }

    public Memo updateToEntity(Long memoId, MemoRequest memoRequest,User user,Item item1){
        return Memo.builder()
                .memoSeq(memoId)
                .content(memoRequest.getContent())
                .item(item1)
                .accessType(memoRequest.getAccessType())
                .user(user)
                .isDeleted(0)
                .build();
    }

    public Memo deleteToEntity(Memo memo,User writeUser, Item item1){
        return Memo.builder()
                .memoSeq(memo.getMemoSeq())
                .isDeleted(1)
                .item(item1)
                .accessType(memo.getAccessType())
                .content(memo.getContent())
                .deletedAt(LocalDateTime.now())
                .user(writeUser)
                .build();
    }

}
