package com.tjjhtjh.memorise.domain.memo.repository.entity;

import com.tjjhtjh.memorise.domain.item.repository.entity.Item;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import com.tjjhtjh.memorise.global.entity.BaseEntity;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.DynamicInsert;
import org.hibernate.annotations.DynamicUpdate;
import org.springframework.data.relational.core.mapping.Table;

import java.io.Serializable;
import java.time.LocalDateTime;

@DynamicUpdate
@DynamicInsert
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
@Entity
@Table(name = "memo")
public class Memo extends BaseEntity implements Serializable {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "memo_seq")
    private Long memoSeq;
    @Column(name = "content")
    private String content;
    @Column(name = "access_type")
    @Enumerated(EnumType.STRING)
    private AccessType accessType;
    @Column(name = "is_deleted")
    private Integer isDeleted;
    @Column(name = "deleted_at")
    private LocalDateTime deletedAt;
    @ManyToOne(targetEntity = User.class ,fetch = FetchType.LAZY)
    @JoinColumn(name = "user_seq" , referencedColumnName = "user_seq")
    private User user;
    @ManyToOne(targetEntity = Item.class ,fetch = FetchType.LAZY)
    @JoinColumn(name = "item_seq" , referencedColumnName = "item_seq")
    private Item item;

}
